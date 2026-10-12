"""Connection/authentication checks plus one real investigation; no GPU claims."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factorymind.core import ROOT, load_assets, predict_cycle, run_investigation
from factorymind.providers import ProviderConfig, check_connection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=213)
    parser.add_argument("--require-amd-evidence", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/provider_probe.json")
    args = parser.parse_args()
    try:
        config = ProviderConfig.from_env()
        health = check_connection(config)
        record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "connection": health, "investigation": None}
        evidence_ok = (not args.require_amd_evidence or (config.provider == "amd_vllm" and
                       health["hardware"]["status"] == "operator_evidence_matched"))
        if health["ready"] and evidence_ok:
            record["investigation"] = run_investigation(predict_cycle(*load_assets(), args.cycle), config=config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2))
        success = record["investigation"] is not None and record["investigation"]["provider_mode"] == "llm"
        print(json.dumps({"ready": health["ready"], "evidence_requirement_met": evidence_ok,
                          "investigation_passed": success, "connection_failure": health["failure"], "evidence": str(args.output)}))
        if not success:
            raise SystemExit(1)
    except Exception as exc:
        raise SystemExit(f"Provider probe failed: {type(exc).__name__}; details omitted.") from None

if __name__ == "__main__":
    main()
