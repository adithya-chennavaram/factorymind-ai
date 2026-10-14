"""Human decisions on blocked out-of-fold Trust Lab evidence.

Deliberately stored separately from the main serving model's `reviews` table:
the same cycle may have a different prediction under each evaluation model.
No actuator connection and no immutable-ledger claim.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reviews import DEFAULT_PATH


def _canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_trust_review(cycle_id: int, fold: int, decision: str,
                      notes: str, evidence: dict[str, Any],
                      path: str | Path = DEFAULT_PATH) -> int:
    """Save an explicit operator decision with its unchanged bounded AMD evidence."""
    if type(cycle_id) is not int or cycle_id < 1 or type(fold) is not int or fold < 1:
        raise ValueError("Invalid cycle or fold")
    if decision not in {"approve_inspection", "reject_inspection"}:
        raise ValueError("Invalid decision")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("Invalid operator note")
    if (not isinstance(evidence, dict)
            or evidence.get("workflow") != "model_trust_lab_blocked_oof"
            or evidence.get("cycle_id") != cycle_id
            or evidence.get("blocked_fold") != fold
            or evidence.get("prediction", {}).get("class_code") != 100
            or evidence.get("amd_hearing", {}).get("provider") != "AMD Token Factory"):
        raise ValueError("Missing or mismatched Trust Lab evidence")
    serialized = _canonical_json(evidence)
    # Hold-out truth is revealed separately; never write it into a pre-review passport.
    forbidden_keys = {"actual", "dataset_label", "true_label", "ground_truth"}
    def contains_hidden_label(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(forbidden_keys & value.keys()) or any(contains_hidden_label(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_hidden_label(v) for v in value)
        return False
    if contains_hidden_label(evidence):
        raise ValueError("Held-out ground truth is not permitted in operator evidence")
    sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(destination)) as connection, connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS trust_lab_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id INTEGER NOT NULL,
            blocked_fold INTEGER NOT NULL,
            decision TEXT NOT NULL,
            notes TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL
        )""")
        cursor = connection.execute("""INSERT INTO trust_lab_reviews
            (cycle_id, blocked_fold, decision, notes, recorded_at_utc, evidence_json, evidence_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (cycle_id, fold, decision, notes,
            datetime.now(timezone.utc).isoformat(), serialized, sha256))
        return int(cursor.lastrowid)


def list_trust_reviews(cycle_id: int, path: str | Path = DEFAULT_PATH) -> list[dict]:
    """Show this workflow's audit history, excluding unrelated main-model reviews."""
    destination = Path(path)
    if not destination.is_file():
        return []
    with closing(sqlite3.connect(destination)) as connection:
        found = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trust_lab_reviews'").fetchone()
        if not found:
            return []
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("""SELECT id, cycle_id, blocked_fold,
            decision, notes, recorded_at_utc, evidence_sha256 FROM trust_lab_reviews
            WHERE cycle_id = ? ORDER BY id DESC LIMIT 20""", (cycle_id,))]


def get_trust_review(review_id: int, path: str | Path = DEFAULT_PATH) -> dict | None:
    """Return one saved passport, including evidence-integrity verification."""
    destination = Path(path)
    if not destination.is_file():
        return None
    with closing(sqlite3.connect(destination)) as connection:
        if not connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trust_lab_reviews'").fetchone():
            return None
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM trust_lab_reviews WHERE id = ?", (review_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["evidence_integrity_ok"] = hashlib.sha256(result["evidence_json"].encode("utf-8")).hexdigest() == result["evidence_sha256"]
    result["evidence"] = json.loads(result.pop("evidence_json"))
    return result
