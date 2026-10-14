"""Trust Lab operator audits must preserve blocked-fold provenance and hide labels."""
import hashlib
import json
import sqlite3

import pytest

from factorymind.reviews import save_review, list_reviews
from factorymind.trust_audit import save_trust_review, list_trust_reviews, get_trust_review
from factorymind.trust_lab import case_payload, trust_evidence_packet
import pandas as pd


def frame():
    return pd.DataFrame([
        {'cycle_id': n, 'fold': 1 if n <= 10 else 2, 'actual': 90 if n in (1, 11) else 100,
         'predicted': 100, 'p_optimal_uncalibrated': .35 + .02 * n,
         'switch_fraction': .02 * (n % 5), 'max_p_optimal_drop': .01 * n,
         'challenge_priority': .5, 'min_stressed_p_optimal': .1,
         'worst_case_sensor': 'PS2', 'worst_case_gain_pct': 2.0, 'scenarios_tested': 50,
         'base_p_optimal': .35 + .02 * n} for n in range(1, 21)])


def fake_report():
    return {
        'selection': {
            'evidence_ids': ['base_score', 'switch_fraction'],
            'action_ids': ['verify_sensor_trace'],
            'verdict': 'review_the_prediction',
        },
        'served_model': 'self-dploy/DeepSeek-V4-Flash',
        'provider': 'AMD Token Factory',
        'limitations': 'Bounded choices, not a causal diagnosis.',
    }


def test_passport_provenance_and_hidden_label():
    record = frame().iloc[0]
    packet = trust_evidence_packet(record, {1}, 20, fake_report(), scores_file_sha256='a' * 64)
    assert packet['blocked_fold'] == 1
    assert packet['prediction']['class_code'] == 100
    assert packet['prediction']['healthy_class_score_uncalibrated'] == pytest.approx(.37)
    assert packet['prediction']['model_context'].startswith('Fold-specific')
    assert packet['evaluation']['selected_for_review'] is True
    assert packet['amd_hearing']['served_model'] == 'self-dploy/DeepSeek-V4-Flash'
    assert 'actual' not in json.dumps(packet)
    assert 'dataset_label' not in json.dumps(packet)


def test_trust_lab_and_serving_decisions_are_separate(tmp_path):
    db = tmp_path / 'reviews.sqlite3'
    original = save_review(1, 'reject_inspection', 'main app', db, evidence={'model':'main'})
    packet = trust_evidence_packet(frame().iloc[0], {1}, 20, fake_report(), scores_file_sha256='b' * 64)
    decision = save_trust_review(1, 1, 'approve_inspection', 'Check signal', packet, db)
    assert original == 1 and decision == 1  # Separate tables intentionally have independent IDs.
    assert list_reviews(1, db)[0]['decision'] == 'reject_inspection'
    history = list_trust_reviews(1, db)
    assert len(history) == 1
    assert history[0]['decision'] == 'approve_inspection'
    saved = get_trust_review(decision, db)
    assert saved['evidence_integrity_ok'] is True
    assert saved['evidence']['prediction']['class_code'] == 100
    assert saved['evidence']['amd_hearing']['selection']['verdict'] == 'review_the_prediction'
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE trust_lab_reviews SET evidence_json='{}' WHERE id=?", (decision,))
    assert get_trust_review(decision, db)['evidence_integrity_ok'] is False


def test_legacy_database_migrates_without_replacing_reviews(tmp_path):
    db = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE reviews (id INTEGER PRIMARY KEY, cycle_id INTEGER, decision TEXT, notes TEXT, recorded_at_utc TEXT)')
        conn.execute("INSERT INTO reviews VALUES (1, 507, 'approve_inspection', 'existing', '2026-10-13')")
    assert list_trust_reviews(507, db) == []
    packet = trust_evidence_packet(frame().iloc[0], {1}, 20, fake_report(), scores_file_sha256='c'*64)
    save_trust_review(1, 1, 'approve_inspection', '', packet, db)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT notes FROM reviews WHERE cycle_id=507').fetchone()[0] == 'existing'


def test_rejects_ground_truth_and_mismatch(tmp_path):
    db = tmp_path / 'reviews.sqlite3'
    packet = trust_evidence_packet(frame().iloc[0], {1}, 20, fake_report(), scores_file_sha256='d' * 64)
    with pytest.raises(ValueError, match='mismatched'):
        save_trust_review(2, 1, 'approve_inspection', '', packet, db)
    packet['actual'] = 90
    with pytest.raises(ValueError, match='ground truth'):
        save_trust_review(1, 1, 'approve_inspection', '', packet, db)
    assert not db.exists()


def test_bad_review_or_unvalidated_hearing(tmp_path):
    record = frame().iloc[0]
    with pytest.raises(ValueError):
        trust_evidence_packet(record, {1}, 25, fake_report(), scores_file_sha256='d'*64)
    with pytest.raises(ValueError):
        trust_evidence_packet(record, {1}, 20, {'provider': 'ollama'}, scores_file_sha256='d'*64)
    packet = trust_evidence_packet(record, {1}, 20, fake_report(), scores_file_sha256='d'*64)
    with pytest.raises(ValueError):
        save_trust_review(1, 1, 'start_machine', '', packet, tmp_path/'reviews.sqlite3')
