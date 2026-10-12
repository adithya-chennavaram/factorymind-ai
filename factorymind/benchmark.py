"""Sequential application benchmarks, with rejected outputs counted as failures."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter
import uuid
import numpy as np
from .core import predict_cycle, investigation_prompt, run_investigation, InvestigationPlan, render_plan
from .providers import check_connection

CASES = [4, 213, 222, 232, 248, 496]


def local_device_inventory():
    result = {"client_os": platform.system(), "client_arch": platform.machine(), "local_gpus": None,
              "note": "Device inventory is not per-request execution proof; client hardware is not remote server hardware."}
    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=10, check=True)
            data = json.loads(proc.stdout)
            result["local_gpus"] = [{"name": d.get("sppci_model"), "cores": d.get("sppci_cores"),
                                      "vram_gib": None, "memory_type": "not_measured"}
                                     for d in data.get("SPDisplaysDataType", [])]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return result


def summarize(records):
    success = [r for r in records if r["provider_mode"] == "llm"]
    times = [r["end_to_end_latency_ms"] for r in success]
    metered = [r for r in success if r["telemetry"]["generated_tokens"] is not None]
    return {"attempts": len(records), "successes": len(success), "failures": len(records) - len(success),
            "failure_categories": dict(Counter(r["failure"] for r in records if r["provider_mode"] != "llm")),
            "success_latency_ms_p50": round(median(times), 3) if times else None,
            "success_latency_ms_p95": round(float(np.percentile(times, 95)), 3) if times else None,
            "generated_tokens": sum(r["telemetry"]["generated_tokens"] for r in metered) if metered else None,
            "usage_reporting_successes": len(metered),
            "tokens_per_second_e2e": round(sum(r["telemetry"]["generated_tokens"] for r in metered) /
                (sum(r["end_to_end_latency_ms"] for r in metered) / 1000), 3) if metered else None}


def run_benchmark(config, assets, *, cycles=CASES, repeats=3, warmups=1):
    if not 1 <= repeats <= 20 or not 0 <= warmups <= 5 or not cycles:
        raise ValueError("Use 1–20 repeats, 0–5 warmups, and at least one cycle")
    model, features = assets
    cases = [predict_cycle(model, features, cycle) for cycle in cycles]
    if any(case["evaluation_split"] != "held_out" for case in cases):
        raise ValueError("Benchmark cases must be held out from the served model")
    messages = [investigation_prompt(case) for case in cases]
    workload_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    health = check_connection(config)
    result = {"schema_version": 1, "run_id": uuid.uuid4().hex,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "provider": config.public(),
              "client_inventory": local_device_inventory(), "connection": health,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ("core.py", "providers.py", "benchmark.py")},
              "workload_sha256": workload_hash, "classifier_sha256": cases[0]["model_version"],
              "cycles": list(cycles), "repeats": repeats, "warmups_requested": warmups,
              "generation": config.generation(),
              "method": "Sequential requests, concurrency 1; application E2E includes prediction, network, generation, schema validation and rendering. Excludes asset load and connection preflight. Tokens are server-reported generated JSON tokens; rate includes prefill/network, not decode-only throughput.",
              "gpu_memory": {"status": "not_measured", "samples": [], "peak_sampled_used_bytes": None},
              "warmup_records": [], "records": [], "status": "preflight_failed"}
    if not health["ready"]:
        result["summary"] = summarize([])
        return result
    if config.provider == "amd_vllm" and health["hardware"]["status"] != "operator_evidence_matched":
        result["status"] = "hardware_evidence_required"
        result["summary"] = summarize([])
        return result
    def invoke(cycle, phase, repeat):
        started_at = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        case = predict_cycle(model, features, cycle)
        investigation = run_investigation(case, config=config)
        elapsed = (perf_counter() - start) * 1000
        tokens = investigation["telemetry"]["generated_tokens"]
        return {"request_id": uuid.uuid4().hex, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "started_at_utc": started_at, "quality": evaluate_quality(investigation),
                "cycle_id": cycle, "phase": phase, "repeat": repeat,
                "end_to_end_latency_ms": round(elapsed, 3),
                "tokens_per_second_e2e": round(tokens / (elapsed / 1000), 3) if tokens is not None else None,
                **investigation}
    for i in range(warmups):
        result["warmup_records"].append(invoke(cycles[i % len(cycles)], "warmup", i + 1))
    # Rotate cases across repeats to reduce consistent first-case ordering effects.
    for repeat in range(repeats):
        order = list(cycles[repeat % len(cycles):]) + list(cycles[:repeat % len(cycles)])
        for cycle in order:
            result["records"].append(invoke(cycle, "measured", repeat + 1))
    result["summary"] = summarize(result["records"])
    result["status"] = "completed" if not result["summary"]["failures"] else "completed_with_failures"
    return result


def compare_runs(left, right):
    if left["status"] not in {"completed", "completed_with_failures"} or right["status"] not in {"completed", "completed_with_failures"}:
        raise ValueError("Both providers need actual completed benchmark runs")
    for key in ("workload_sha256", "classifier_sha256", "cycles", "repeats", "warmups_requested", "generation"):
        if left[key] != right[key]:
            raise ValueError(f"Incomparable benchmark protocol: {key}")
    return {"note": "Observed system comparison. Different model weights, quantization, tokenizers, server versions and hardware are confounded; this does not isolate GPU speedup.",
            "runs": [{"run_id": r["run_id"], "provider": r["provider"], "summary": r["summary"]} for r in (left, right)],
            "by_cycle": [{"cycle_id": cycle, "left": summarize([r for r in left["records"] if r["cycle_id"] == cycle]),
                          "right": summarize([r for r in right["records"] if r["cycle_id"] == cycle])} for cycle in left["cycles"]]}


def evaluate_quality(result):
    """Mechanical contract checks, not an expert judgment of diagnostic usefulness."""
    grounded = False
    try:
        plan = InvestigationPlan.model_validate(result["telemetry"]["validated_plan"])
        grounded = result["investigation"] == render_plan(plan, result["case"])
    except (KeyError, ValueError, TypeError):
        pass
    return {"accepted_structured_response": result["provider_mode"] == "llm",
            "exact_evidence_catalog_render": grounded,
            "human_approval_required": result.get("requires_human_approval") is True,
            "equipment_control_absent": result.get("equipment_control") is False,
            "expert_usefulness_review": "pending",
            "scope": "Mechanical grounding/approval checks do not establish causal diagnosis or maintenance usefulness."}


def attach_gpu_telemetry(run, telemetry):
    """Attach actual overlapping host samples; reject mismatched/stale evidence."""
    hardware = run["connection"]["hardware"].get("gpu_details")
    if run["provider"]["requested_provider"] != "amd_vllm" or not hardware:
        raise ValueError("Only an actual AMD-profile run with hardware evidence can receive GPU telemetry")
    if telemetry.get("deployment_id") != hardware["deployment_id"] or telemetry.get("schema_version") != 1:
        raise ValueError("GPU telemetry deployment mismatch")
    records = run["records"]
    start = min(datetime.fromisoformat(r["started_at_utc"]) for r in records)
    end = max(datetime.fromisoformat(r["recorded_at_utc"]) for r in records)
    samples = [s for s in telemetry["samples"] if start <= datetime.fromisoformat(s["recorded_at_utc"]) <= end]
    measured = []
    for sample in samples:
        rows = sample.get("vram", [])
        if len(rows) != 1:
            continue
        row = rows[0]
        used, total = row.get("used_bytes"), row.get("total_bytes")
        if isinstance(used, int) and isinstance(total, int) and 0 <= used <= total and total > 0:
            measured.append(used)
    if not measured:
        raise ValueError("No overlapping single-GPU byte measurements; verify clocks and sysfs availability")
    run["gpu_memory"] = {"status": "operator_samples_matched", "samples": samples,
        "peak_sampled_used_bytes": max(measured), "minimum_sampled_used_bytes": min(measured),
        "note": "Host-wide sampled usage, not per-request allocation or exact peak. Operator must verify clocks and mapping. No cryptographic attestation."}
    return run
