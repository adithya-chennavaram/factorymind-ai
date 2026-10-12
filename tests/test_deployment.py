from datetime import datetime, timezone
import pytest
from factorymind.providers import HardwareEvidence
from factorymind.deployment import deployment_plan
from scripts.collect_amd_hardware import collect


def inventory(free, total=192, arch="gfx942", name="AMD Instinct MI300X SYNTHETIC FIXTURE"):
    return HardwareEvidence(collected_at_utc=datetime.now(timezone.utc), deployment_id="a" * 32,
        image_digest="vllm/vllm-openai-rocm@sha256:" + "b" * 64, hip_version="synthetic", torch_version="synthetic",
        vllm_version="synthetic", gpu_compute_check=True,
        devices=[dict(name=name, architecture=arch, free_vram_gib=free, total_vram_gib=total)])


def test_model_selection_uses_free_memory():
    assert deployment_plan(inventory(180))["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert deployment_plan(inventory(16))["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ValueError): deployment_plan(inventory(4))
    with pytest.raises(ValueError): deployment_plan(inventory(300))
    with pytest.raises(ValueError): deployment_plan(inventory(180, arch="gfx1100", name="AMD Radeon fixture"))
    assert deployment_plan(inventory(16))["status"] == "prepared_only_not_deployed"


def test_no_cpu_or_cuda_evidence():
    from types import SimpleNamespace
    cpu = SimpleNamespace(version=SimpleNamespace(hip=None))
    with pytest.raises(ValueError): collect(cpu, "unused", "a" * 32, "synthetic")


def test_probe_failure_blocks_plan():
    e = inventory(180); e.gpu_compute_check = False
    with pytest.raises(ValueError): deployment_plan(e)


def test_multiple_gpu_allocation_refused():
    e = inventory(180)
    e.devices = e.devices * 8
    with pytest.raises(ValueError, match='Exactly one'): deployment_plan(e)
