"""Local human inspection audit. Never actuates equipment."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data/reviews.sqlite3"


def save_review(cycle_id: int, decision: str, notes: str = "", path: str | Path = DEFAULT_PATH,
                *, evidence: dict | None = None) -> int:
    if decision not in {"approve_inspection", "reject_inspection"} or cycle_id < 1:
        raise ValueError("Invalid review decision or cycle")
    if len(notes) > 2000:
        raise ValueError("Notes must not exceed 2,000 characters")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(destination)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                           "cycle_id INTEGER, decision TEXT, notes TEXT, recorded_at_utc TEXT)")
        columns = {r[1] for r in connection.execute("PRAGMA table_info(reviews)")}
        if "evidence_json" not in columns:
            connection.execute("ALTER TABLE reviews ADD COLUMN evidence_json TEXT")
        result = connection.execute(
            "INSERT INTO reviews (cycle_id, decision, notes, recorded_at_utc, evidence_json) VALUES (?, ?, ?, ?, ?)",
            (cycle_id, decision, notes, datetime.now(timezone.utc).isoformat(), json.dumps(evidence or {})))
        return result.lastrowid


def list_reviews(cycle_id: int, path: str | Path = DEFAULT_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT id, cycle_id, decision, notes, recorded_at_utc FROM reviews WHERE cycle_id=? ORDER BY id DESC LIMIT 20",
            (cycle_id,))]
