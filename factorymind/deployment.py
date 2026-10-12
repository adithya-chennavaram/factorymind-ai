"""Offline deployment planning from observed hardware. Never provisions anything."""
from datetime import datetime, timezone
from .providers import HardwareEvidence

# Intentionally restricted to the documented Instinct families. A successful
# compute probe is necessary, but an actual model launch is still a separate gate.
INSTINCT_ARCHES = {"gfx90a", "gfx942", "gfx950"}


def deployment_plan(evidence: HardwareEvidence, gpu_index: int = 0) -> dict:
    age = (datetime.now(timezone.utc) - evidence.collected_at_utc).total_seconds()
    if evidence.schema_version != 1 or not 0 <= age <= 86400 or not evidence.gpu_compute_check:
        raise ValueError("Fresh successful GPU compute evidence is required")
    if len(evidence.devices) != 1 or gpu_index != 0:
        raise ValueError("Exactly one allocated GPU is required; multi-GPU configurations are refused")
    if not 0 <= gpu_index < len(evidence.devices):
        raise ValueError("Requested GPU index was not observed")
    gpu = evidence.devices[gpu_index]
    if gpu.architecture not in INSTINCT_ARCHES or "AMD" not in gpu.name.upper() or "MI" not in gpu.name.upper():
        raise ValueError("GPU requires manual compatibility review; no automatic image/model selection")
    if gpu.free_vram_gib > gpu.total_vram_gib:
        raise ValueError("Invalid free VRAM")
    # Conservative project admission budgets, NOT vendor minima or measured footprints.
    budget = gpu.free_vram_gib * .75
    if budget >= 24:
        model, reserved = "Qwen/Qwen2.5-7B-Instruct", 24
    elif budget >= 8:
        model, reserved = "Qwen/Qwen2.5-1.5B-Instruct", 8
    else:
        raise ValueError("Insufficient observed free VRAM for this validated planning policy")
    return {"schema_version": 1, "deployment_id": evidence.deployment_id,
            "gpu_index": gpu_index, "gpu": gpu.model_dump(), "image_digest": evidence.image_digest,
            "selection_reason": "Preferred 7B model fits conservative budget" if reserved == 24 else "Preferred 7B exceeds observed free-memory admission budget; smaller fallback requires actual loading validation",
            "model": model, "model_revision": None, "served_model_name": "factorymind-" + evidence.deployment_id,
            "dtype": "half", "max_model_len": 4096, "max_num_seqs": 2, "tensor_parallel_size": 1,
            "gpu_memory_utilization": round(min(.8, budget / gpu.total_vram_gib), 3),
            "planning_budget_gib": budget, "estimated_admission_budget_gib": reserved,
            "status": "prepared_only_not_deployed",
            "limits": "Admission budgets are conservative estimates. Pin model revision, verify driver compatibility and actual vLLM model loading/memory on the host before inference. No runtime performance claim."}
