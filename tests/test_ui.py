"""Streamlit flow on real artifacts when locally available; no live LLM needed."""
from pathlib import Path
import sqlite3
import pytest
from streamlit.testing.v1 import AppTest
from factorymind import core

ROOT = Path(__file__).resolve().parent.parent

@pytest.mark.skipif(not (ROOT / "artifacts/predictions.csv").exists(), reason="Download and train real UCI artifacts first")
def test_real_data_ui_flow(monkeypatch, tmp_path):
    db = tmp_path / "ui-reviews.sqlite3"
    monkeypatch.setenv("FACTORYMIND_REVIEW_DB", str(db))
    def unavailable(case, **kwargs):
        raise TimeoutError("simulated offline provider")
    monkeypatch.setattr(core, "investigate", unavailable)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.metric[0].value in ("Small lag", "Severe lag", "Near failure")
    app.button[0].click().run()
    assert not app.exception
    assert any("Rule-based fallback" in msg.value for msg in app.warning)
    app.text_input[0].set_value("AUTOMATED UI TEST")
    next(b for b in app.button if b.label == "Approve inspection review").click().run()
    assert not app.exception
    assert any("recorded" in msg.value for msg in app.success)
    next(b for b in app.button if b.label == "Reject inspection review").click().run()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT decision FROM reviews ORDER BY id").fetchall() == [("approve_inspection",), ("reject_inspection",)]
        assert 'fallback' in conn.execute("SELECT evidence_json FROM reviews LIMIT 1").fetchone()[0]
    next(s for s in app.selectbox if s.label == "Case collection").set_value("Held-out · model errors").run()
    assert not app.exception
    assert not any(b.label == "Approve inspection review" for b in app.button)
    assert any("incorrect" in msg.value for msg in app.markdown)
