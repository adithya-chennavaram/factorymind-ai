import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import amd_host


def test_sysfs_memory_measured_bytes_and_missing_not_zero(tmp_path):
    assert amd_host.memory_snapshot(tmp_path) == []
    device = tmp_path / 'card0/device'; device.mkdir(parents=True)
    for key, value in {'vendor': '0x1002', 'mem_info_vram_used': '1024', 'mem_info_vram_total': '8192'}.items():
        (device / key).write_text(value)
    assert amd_host.memory_snapshot(tmp_path) == [{'card': 'card0', 'used_bytes': 1024, 'total_bytes': 8192, 'free_bytes': 7168}]
    (device / 'mem_info_vram_used').write_text('9000')
    assert amd_host.memory_snapshot(tmp_path) == []


def test_start_requires_authorized_host_and_key(monkeypatch):
    monkeypatch.delenv('VLLM_API_KEY', raising=False)
    with pytest.raises(ValueError): amd_host.lifecycle('start', Path('fixture'))
    with pytest.raises(ValueError): amd_host.lifecycle('start', Path('fixture'), True)


def test_failed_start_cleans_container_and_hides_output(monkeypatch):
    monkeypatch.setenv('VLLM_API_KEY', 'synthetic-test-key-123456')
    calls = []
    def run(command, timeout):
        calls.append(command)
        return {'ok': 'down' in command, 'stdout': 'potential sensitive diagnostic'}
    monkeypatch.setattr(amd_host, 'observed', run)
    result = amd_host.lifecycle('start', Path('fixture'), True)
    assert not result['ok'] and result['cleanup_ok'] and result['stdout'] is None
    assert '--wait' in calls[0] and 'down' in calls[1]
    assert '127.0.0.1:8000:8000' in Path('deploy/compose.amd.yaml').read_text()


def test_monitor_bad_bounds_never_runs_commands(monkeypatch):
    monkeypatch.setattr(amd_host, 'observed', lambda *a, **k: pytest.fail('unexpected command'))
    with pytest.raises(ValueError): amd_host.monitor('bad', 1, 1)
    with pytest.raises(ValueError): amd_host.monitor('a' * 32, 10000, 1)
