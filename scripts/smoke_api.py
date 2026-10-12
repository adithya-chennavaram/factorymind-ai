"""Live local integration smoke. Creates clearly marked test audit records."""
import json
from pathlib import Path
import httpx


def main():
    with httpx.Client(base_url="http://127.0.0.1:8001", timeout=65) as client:
        health = client.get("/health")
        health.raise_for_status()
        results = {"health": health.json(), "cases": []}
        for cycle in [4, 213, 248]:
            result = client.post("/investigate", json={"cycle_id": cycle})
            result.raise_for_status()
            results["cases"].append(result.json())
        trace = client.get("/sensors/213/PS1")
        trace.raise_for_status()
        results["trace_samples"] = trace.json()["sample_count"]
        for decision in ["approve_inspection", "reject_inspection"]:
            r = client.post("/reviews", json={"cycle_id": 213, "decision": decision,
                "notes": "AUTOMATED API SMOKE TEST — no real inspection requested"})
            r.raise_for_status()
        results["audit"] = client.get("/reviews/213").json()
    destination = Path(__file__).resolve().parent.parent / "artifacts/local_smoke.json"
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps({"health": results["health"], "investigations": [
        {"cycle": c["case"]["cycle_id"], "mode": c["provider_mode"], "latency_ms": c["latency_ms"]} for c in results["cases"]],
        "audit_rows": len(results["audit"]["reviews"]), "trace_samples": results["trace_samples"]}, indent=2))

if __name__ == "__main__":
    main()
