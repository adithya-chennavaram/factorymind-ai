"""Read-only, reproducible retrospective audit of blocked out-of-fold scores.

The utility is deliberately separate from model serving: per-fold Random Forest
predictions here are NOT outputs of the random-split model served by app.py.
Never pass dataset labels to an LLM or use them in ranking.
"""
from __future__ import annotations

import json
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {
    'cycle_id', 'fold', 'actual', 'predicted', 'p_optimal_uncalibrated',
    'switch_fraction', 'max_p_optimal_drop', 'challenge_priority',
    'min_stressed_p_optimal', 'base_p_optimal', 'worst_case_sensor', 'worst_case_gain_pct',
    'scenarios_tested',
}
METHODS = {
    'Confidence-only': ('p_optimal_uncalibrated', True),
    'Switch-fraction (exploratory)': ('switch_fraction', False),
    'Maximum drop (failed hypothesis)': ('max_p_optimal_drop', False),
    'Original combined (failed hypothesis)': ('challenge_priority', False),
}
SENSOR_IDS = {'PS1', 'PS2', 'FS1', 'TS1', 'VS1'}


def load_scores(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError('Missing score columns: ' + ', '.join(sorted(missing)))
    if frame.empty or frame.cycle_id.isna().any() or frame.cycle_id.duplicated().any():
        raise ValueError('Expected nonempty unique cycle IDs')
    if (frame.predicted != 100).any():
        raise ValueError('Only out-of-fold predicted-optimal cases can be audited')
    if frame.fold.isna().any() or frame.actual.isna().any():
        raise ValueError('Missing evaluation data')
    for key in ['p_optimal_uncalibrated','switch_fraction','max_p_optimal_drop',
                'challenge_priority','min_stressed_p_optimal']:
        data = frame[key].to_numpy(dtype=float)
        if not np.isfinite(data).all() or (data < 0).any() or (data > 1).any():
            raise ValueError(f'Invalid values in {key}')
    if not frame.worst_case_sensor.isin(SENSOR_IDS).all() or (frame.scenarios_tested <= 0).any():
        raise ValueError('Malformed stress-test metadata')
    return frame.sort_values('cycle_id').reset_index(drop=True)


def select_for_review(frame: pd.DataFrame, method: str, budget: int) -> set[int]:
    """Deterministic per-fold selection; no ground truth accessed."""
    if method not in METHODS or budget not in (10, 20, 30):
        raise ValueError('Unsupported method or review budget')
    field, ascending = METHODS[method]
    selected = set()
    for _, group in frame.groupby('fold', sort=True):
        k = floor(len(group) * budget / 100)
        ordered = group.sort_values([field, 'cycle_id'], ascending=[ascending, True], kind='stable')
        selected.update(int(v) for v in ordered.cycle_id.head(k))
    return selected


def comparison(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Labels are consulted only after all review selections are determined."""
    false_ids = set(frame.loc[frame.actual != 100, 'cycle_id'].astype(int))
    rows = []
    for method in METHODS:
        selected = select_for_review(frame, method, budget)
        caught = len(selected & false_ids)
        rows.append({'Review policy': method, 'Cycles reviewed': len(selected),
                     'False-green caught': caught, 'False-green missed': len(false_ids) - caught,
                     'Catch rate': caught / len(false_ids) if false_ids else 0.0,
                     'Inspection precision': caught / len(selected) if selected else 0.0})
    return pd.DataFrame(rows)


def case_payload(record: pd.Series, selected: set[int]) -> dict[str, Any]:
    """Safe case packet: no ground truth, cycle ID, or curated example labels."""
    return {
        'original_prediction': 'Optimal switching',
        'healthy_class_score_uncalibrated': round(float(record.p_optimal_uncalibrated), 4),
        'review_selected_by_confidence_only': int(record.cycle_id) in selected,
        'simulated_stress': {
            'scenario_count': int(record.scenarios_tested),
            'fraction_changing_predicted_class': round(float(record.switch_fraction), 4),
            'maximum_decrease_in_healthy_class_score': round(float(record.max_p_optimal_drop), 4),
            'lowest_healthy_class_score': round(float(record.min_stressed_p_optimal), 4),
            'sensor_with_largest_score_drop': str(record.worst_case_sensor),
            'sensor_gain_at_largest_score_drop_pct': round(float(record.worst_case_gain_pct), 3),
        },
        'limits': 'Simulated sensor gains are not physical fault progression, sensor-certified noise, calibrated probabilities, or prospective validation.',
    }


def request_amd_hearing(payload: dict[str, Any], config=None) -> dict[str, Any]:
    """Token Factory makes bounded, evidence-based selections only.

    All numerical descriptions are rendered locally from verified data.
    The held-out label is NEVER provided to the model.
    """
    from factorymind.providers import ProviderConfig
    config = config or ProviderConfig.from_env()
    if config.provider != 'amd_tokenfactory':
        raise ValueError('Set LLM_PROVIDER=amd_tokenfactory for this hearing')
    system = (
        'You are an industrial ML quality reviewer, NOT a machine diagnostician. '
        'Use ONLY the input JSON to choose evidence and human checks. '
        'Respond with JSON object keys evidence_ids (1-3 unique IDs), action_ids (1-3 unique IDs), '
        'and verdict (one of review_the_prediction, examine_robustness, no_extra_claims). '
        'Allowed evidence_ids: base_score, review_policy, switch_fraction, max_score_drop, worst_gain. '
        'Allowed action_ids: verify_sensor_trace, crosscheck_independent_measurement, review_model_limitations. '
        'No new measurements, equipment commands, diagnoses, or causal conclusions.'
    )
    with config.client() as client:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user', 'content': json.dumps(payload, sort_keys=True)}],
            temperature=config.temperature, top_p=config.top_p,
            max_tokens=config.max_tokens, response_format={'type': 'json_object'},
        )
    actual_model = getattr(response, 'model', None)
    if not config.matches_model(actual_model):
        raise ValueError('Unexpected AMD response model')
    raw = json.loads(response.choices[0].message.content)
    if not isinstance(raw, dict) or set(raw) != {'evidence_ids', 'action_ids', 'verdict'}:
        raise ValueError('Unexpected AMD review schema')
    allowed_evidence = {'base_score', 'review_policy', 'switch_fraction', 'max_score_drop', 'worst_gain'}
    allowed_actions = {'verify_sensor_trace', 'crosscheck_independent_measurement', 'review_model_limitations'}
    for field, allowed in [('evidence_ids', allowed_evidence), ('action_ids', allowed_actions)]:
        values = raw[field]
        if not isinstance(values, list) or not 1 <= len(values) <= 3 or len(values) != len(set(map(str, values))) or not all(type(v) is str and v in allowed for v in values):
            raise ValueError('Invalid AMD ' + field)
    if raw['verdict'] not in {'review_the_prediction', 'examine_robustness', 'no_extra_claims'}:
        raise ValueError('Invalid AMD review verdict')
    return {'selection': raw, 'served_model': actual_model,
            'provider': 'AMD Token Factory',
            'limitations': 'Bounded choices, not a causal diagnosis or a machine-control action.'}


def render_hearing(report: dict, payload: dict) -> dict[str, Any]:
    """All numbers are sourced from the local evidence packet, not the LLM."""
    stress = payload['simulated_stress']
    evidence = {
        'base_score': f"Original uncalibrated healthy-class score: {payload['healthy_class_score_uncalibrated']:.1%}.",
        'review_policy': 'Confidence-only review: ' + ('selected for review.' if payload['review_selected_by_confidence_only'] else 'not selected at this budget.'),
        'switch_fraction': f"Prediction changed in {stress['fraction_changing_predicted_class']:.1%} of simulated gain scenarios.",
        'max_score_drop': f"Largest simulated drop in healthy-class score: {stress['maximum_decrease_in_healthy_class_score']:.1%} points.",
        'worst_gain': f"Largest drop was associated with a simulated {stress['sensor_gain_at_largest_score_drop_pct']:+g}% gain on {stress['sensor_with_largest_score_drop']}.",
    }
    actions = {
        'verify_sensor_trace': 'Review the recorded sensor trace and acquisition quality.',
        'crosscheck_independent_measurement': 'Ask a qualified operator to verify the reading with an independent measurement if practical.',
        'review_model_limitations': 'Review blocked-validation performance and the model’s known limitations.',
    }
    return {
        'verdict': report['selection']['verdict'].replace('_', ' ').capitalize(),
        'evidence': [evidence[k] for k in report['selection']['evidence_ids']],
        'checks': [actions[k] for k in report['selection']['action_ids']],
        'provider': report['provider'], 'served_model': report['served_model'],
        'limitations': report['limitations'],
    }


def trust_evidence_packet(record: pd.Series, selected: set[int], budget: int,
                          report: dict[str, Any], *, scores_file_sha256: str) -> dict[str, Any]:
    """Create a provenance-rich decision packet without ground truth or any API key.

    The original row is the blocked *out-of-fold* model result, NOT the main
    application's separately trained demonstration model. Floats are unrounded.
    """
    if budget not in (10, 20, 30):
        raise ValueError('Invalid per-fold review budget')
    if report.get('provider') != 'AMD Token Factory' or not isinstance(report.get('selection'), dict):
        raise ValueError('A validated AMD Token Factory hearing is required')
    if not isinstance(scores_file_sha256, str) or len(scores_file_sha256) != 64 or any(c not in '0123456789abcdef' for c in scores_file_sha256):
        raise ValueError('A SHA-256 digest of the scored CSV is required')
    record_values = {
        'p_optimal_uncalibrated': float(record.p_optimal_uncalibrated),
        'base_p_optimal': float(record.base_p_optimal),
        'min_stressed_p_optimal': float(record.min_stressed_p_optimal),
        'max_p_optimal_drop': float(record.max_p_optimal_drop),
        'switch_fraction': float(record.switch_fraction),
        'worst_case_sensor': str(record.worst_case_sensor),
        'worst_case_gain_pct': float(record.worst_case_gain_pct),
        'scenarios_tested': int(record.scenarios_tested),
    }
    packet = {
        'schema_version': 1,
        'workflow': 'model_trust_lab_blocked_oof',
        'cycle_id': int(record.cycle_id),
        'blocked_fold': int(record.fold),
        'prediction': {
            'class_code': int(record.predicted),
            'class_label': 'Optimal switching',
            'healthy_class_score_uncalibrated': float(record.p_optimal_uncalibrated),
            'model_context': 'Fold-specific blocked out-of-fold Random Forest; NOT main dashboard model',
            'fold_model_sha256': None,  # Not recorded by the evaluation script; do not invent it.
        },
        'evaluation': {
            'review_policy': 'Confidence-only (lowest uncalibrated healthy-class score)',
            'review_budget_pct_predicted_green_per_fold': budget,
            'selected_for_review': int(record.cycle_id) in selected,
            'scores_source': 'artifacts/green_challenge_scored.csv',
            'scores_file_sha256': scores_file_sha256,
        },
        'stress_test': record_values,
        'amd_hearing': {
            'provider': report['provider'],
            'served_model': str(report['served_model']),
            'selection': report['selection'],
            'rendered': render_hearing(report, case_payload(record, selected)),
            'limitations': str(report['limitations']),
        },
        'ground_truth_included_in_review': False,
        'limitations': 'Experimental UCI rig; hypothesis only. Stress tests are simulated gain changes, not prospective reliability guarantees. No equipment control.',
    }
    return packet
