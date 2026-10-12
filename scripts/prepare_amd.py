"""Generate a deployment plan from actual hardware JSON without network or Docker."""
import argparse
import json
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factorymind.providers import HardwareEvidence
from factorymind.deployment import deployment_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/amd_plan.json"))
    parser.add_argument("--model-revision", help="Reviewed Hugging Face model commit (40 hex characters)")
    parser.add_argument("--compose-env", type=Path, help="Write non-secret Docker Compose variables; requires --model-revision")
    args = parser.parse_args()
    try:
        plan = deployment_plan(HardwareEvidence.model_validate_json(args.hardware.read_text()), args.gpu_index)
        if args.model_revision and not re.fullmatch(r"[a-f0-9]{40}", args.model_revision):
            raise ValueError("An immutable model revision is required")
        if args.compose_env and not args.model_revision:
            raise ValueError("Compose variables require a pinned model revision")
        plan["model_revision"] = args.model_revision
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2))
        if args.compose_env:
            args.compose_env.parent.mkdir(parents=True, exist_ok=True)
            values = {"VLLM_IMAGE": plan["image_digest"], "AMD_GPU_INDEX": plan["gpu_index"],
                      "VLLM_MODEL": plan["model"], "VLLM_MODEL_REVISION": args.model_revision,
                      "VLLM_SERVED_MODEL": plan["served_model_name"],
                      "VLLM_MEMORY_UTILIZATION": plan["gpu_memory_utilization"]}
            args.compose_env.write_text("# Prepared from observed hardware. Contains no API key.\n" +
                                       "".join(f"{key}={value}\n" for key, value in values.items()))
        print(f"Prepared only: {plan['model']} on observed {plan['gpu']['name']}; plan: {args.output}")
    except (OSError, ValueError, TypeError):
        raise SystemExit("Plan refused: provide fresh, valid Instinct GPU/VRAM/ROCm compute evidence from the candidate image.") from None

if __name__ == "__main__":
    main()
