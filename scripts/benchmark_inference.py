"""Benchmark the configured provider, or compare two already measured runs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factorymind.benchmark import run_benchmark, compare_runs
from factorymind.core import load_assets
from factorymind.providers import ProviderConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    try:
        if args.compare:
            result = compare_runs(*(json.loads(p.read_text()) for p in args.compare))
        else:
            result = run_benchmark(ProviderConfig.from_env(), load_assets(), repeats=args.repeats, warmups=args.warmups)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result.get("summary", result), indent=2))
        print(f"Evidence saved to {args.output}; status: {result.get('status', 'comparison')}")
        if result.get("status", "completed") != "completed":
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Benchmark could not run ({type(exc).__name__}); check configuration, assets and matching benchmark protocols.") from None

if __name__ == "__main__":
    main()
