from pathlib import Path

import pytest

from factorymind.bootstrap import BootstrapError, ensure_assets


def test_bootstrap_is_noop_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTORYMIND_AUTO_SETUP", raising=False)
    ensure_assets(tmp_path)
    assert not (tmp_path / ".factorymind_bootstrap.lock").exists()


def test_bootstrap_failure_is_clear_and_cleans_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORYMIND_AUTO_SETUP", "1")
    monkeypatch.setattr("factorymind.bootstrap.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("network")))
    with pytest.raises(BootstrapError, match="initialization failed"):
        ensure_assets(tmp_path)
    assert not (tmp_path / ".factorymind_bootstrap.lock").exists()
