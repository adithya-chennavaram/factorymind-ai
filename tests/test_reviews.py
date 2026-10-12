import sqlite3
import pytest
from factorymind.reviews import save_review


def test_review_audit(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    save_review(4, "approve_inspection", "Review required", path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT cycle_id, decision FROM reviews").fetchone() == (4, "approve_inspection")
    with pytest.raises(ValueError):
        save_review(4, "start_machine", path=path)
