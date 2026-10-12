"""Run INSIDE a candidate ROCm vLLM image on the approved AMD host.

Does not install, download, provision or start any server. Prints observed GPU
properties and a small GPU compute check. Does not capture environment/secrets.
"""
import argparse
from datetime import datetime, timezone
import json
import re
import uuid
import sys
from contextlib import redirect_stdout


def collect(torch, image_digest, deployment_id, vllm_version):
    if not torch.version.hip or not torch.cuda.is_available():
        raise ValueError("A working HIP GPU runtime is required; CPU/CUDA is not AMD evidence")
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible GPU is required; do not mask an eight-GPU allocation")
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        arch = prop.gcnArchName.split(":")[0]
        if "AMD" not in prop.name.upper() or not re.fullmatch(r"gfx[0-9a-f]+", arch):
            raise ValueError("Unexpected GPU identity")
        # Verify actual GPU computation inside the selected image, not just a device name.
        value = torch.ones((32, 32), device=f"cuda:{index}")
        product = value @ value
        torch.cuda.synchronize(index)
        if product[0, 0].item() != 32:
            raise ValueError("GPU compute validation failed")
        devices.append({"name": prop.name, "architecture": arch,
                        "total_vram_gib": round(total / 2**30, 3), "free_vram_gib": round(free / 2**30, 3)})
    if not devices:
        raise ValueError("No GPU devices observed")
    return {"schema_version": 1, "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "deployment_id": deployment_id, "image_digest": image_digest,
            "hip_version": torch.version.hip, "torch_version": torch.__version__,
            "vllm_version": vllm_version, "gpu_compute_check": True, "devices": devices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--deployment-id", default=uuid.uuid4().hex)
    args = parser.parse_args()
    if not re.fullmatch(r"vllm/vllm-openai-rocm@sha256:[a-f0-9]{64}", args.image_digest):
        parser.error("An immutable official ROCm image digest is required")
    if not re.fullmatch(r"[a-f0-9]{32}", args.deployment_id):
        parser.error("Invalid deployment ID")
    try:
        with redirect_stdout(sys.stderr):
            import torch
            import vllm
            evidence = collect(torch, args.image_digest, args.deployment_id, vllm.__version__)
        print(json.dumps(evidence, indent=2))
    except Exception as exc:
        # No driver logs/hostnames or sensitive exception strings are exported.
        raise SystemExit(f"AMD hardware collection failed ({type(exc).__name__}); inspect the approved host locally.") from None


if __name__ == "__main__":
    main()
