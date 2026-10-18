import pytest
from factorymind.benchmark import summarize, compare_runs


def record(latency, tokens, mode="llm"):
    return {"end_to_end_latency_ms": latency, "telemetry": {"generated_tokens": tokens},
            "provider_mode": mode, "failure": "invalid_response" if mode == "fallback" else None}


def test_counts_tokens_and_failures_without_invented_usage():
    report = summarize([record(1000, 20), record(2000, 40), record(300, 10, "fallback"), record(900, None)])
    assert report["attempts"] == 4 and report["successes"] == 3 and report["failures"] == 1
    assert report["generated_tokens"] == 60
    assert report["tokens_per_second_e2e"] == 20
    assert report["usage_reporting_successes"] == 2
    assert summarize([record(100, None)])["generated_tokens"] is None
    assert summarize([])["success_latency_ms_p50"] is None


def test_missing_run_cannot_be_compared():
    with pytest.raises(ValueError): compare_runs({"status": "preflight_failed"}, {"status": "completed"})


def test_different_workloads_cannot_be_compared():
    with pytest.raises(ValueError):
        compare_runs({"status": "completed", "workload_sha256": "one"}, {"status": "completed", "workload_sha256": "two"})


def test_failed_preflight_performs_no_inference(monkeypatch):
    from factorymind import benchmark
    from factorymind.providers import ProviderConfig
    case = {"evaluation_split": "held_out", "model_version": "test"}
    monkeypatch.setattr(benchmark, "predict_cycle", lambda *args: case)
    monkeypatch.setattr(benchmark, "investigation_prompt", lambda case: [])
    monkeypatch.setattr(benchmark, "local_device_inventory", lambda: {})
    monkeypatch.setattr(benchmark, "check_connection", lambda config: {"ready": False})
    def forbidden(*args, **kwargs): raise AssertionError("Must not infer after failed checks")
    monkeypatch.setattr(benchmark, "run_investigation", forbidden)
    result = benchmark.run_benchmark(ProviderConfig("amd_tokenfactory", "https://developer.amd.com.cn/radeon/api/v1", "DeepSeek-V4-Flash", "rc-" + "a" * 48), (None, None))
    assert result["status"] == "preflight_failed" and result["summary"]["attempts"] == 0
