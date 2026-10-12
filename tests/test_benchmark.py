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
    result = benchmark.run_benchmark(ProviderConfig("ollama", "http://localhost:11434/v1", "llama3.2", "ollama"), (None, None))
    assert result["status"] == "preflight_failed" and result["summary"]["attempts"] == 0


def test_amd_benchmark_requires_matching_hardware(monkeypatch):
    from factorymind import benchmark
    from factorymind.providers import ProviderConfig
    monkeypatch.setattr(benchmark, "predict_cycle", lambda *args: {"evaluation_split": "held_out", "model_version": "test"})
    monkeypatch.setattr(benchmark, "investigation_prompt", lambda case: [])
    monkeypatch.setattr(benchmark, "local_device_inventory", lambda: {})
    monkeypatch.setattr(benchmark, "check_connection", lambda config: {"ready": True, "hardware": {"status": "unverified"}})
    def forbidden(*args, **kwargs): raise AssertionError("No AMD measurement without actual hardware record")
    monkeypatch.setattr(benchmark, "run_investigation", forbidden)
    result = benchmark.run_benchmark(ProviderConfig("amd_vllm", "http://localhost:8000/v1", "some-model", "private-test-key-123456"), (None, None))
    assert result["status"] == "hardware_evidence_required"
    assert result["records"] == []


def test_gpu_telemetry_requires_identity_and_overlapping_measurements():
    from factorymind.benchmark import attach_gpu_telemetry
    run = {'provider': {'requested_provider': 'amd_vllm'}, 'connection': {'hardware': {'gpu_details': {'deployment_id': 'a' * 32}}},
           'records': [{'started_at_utc': '2026-01-01T00:00:00+00:00', 'recorded_at_utc': '2026-01-01T00:00:02+00:00'}]}
    sample = {'recorded_at_utc': '2026-01-01T00:00:01+00:00', 'vram': [{'used_bytes': 1024, 'total_bytes': 2048}]}
    telemetry = {'schema_version': 1, 'deployment_id': 'b' * 32, 'samples': [sample]}
    with pytest.raises(ValueError): attach_gpu_telemetry(run, telemetry)
    telemetry['deployment_id'] = 'a' * 32
    assert attach_gpu_telemetry(run, telemetry)['gpu_memory']['peak_sampled_used_bytes'] == 1024
    sample['recorded_at_utc'] = '2025-01-01T00:00:01+00:00'
    with pytest.raises(ValueError): attach_gpu_telemetry(run, telemetry)


def test_quality_never_equates_fallback_with_grounded_ai():
    from factorymind.benchmark import evaluate_quality
    quality = evaluate_quality({'provider_mode': 'fallback', 'telemetry': {}, 'requires_human_approval': True, 'equipment_control': False})
    assert not quality['accepted_structured_response'] and not quality['exact_evidence_catalog_render']
    assert quality['expert_usefulness_review'] == 'pending'
