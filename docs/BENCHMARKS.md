# Managed inference benchmarks

Configure AMD Token Factory using [the setup guide](AMD_TOKEN_FACTORY_ACTIII.md). Benchmark runs consume team quota; run only when authorized.

```bash
python scripts/benchmark_inference.py --repeats 1 --warmups 0 --output artifacts/amd_benchmark.json
```

The default workload contains six held-out cases. Records include requested and returned model, actual provider, generation settings, end-to-end latency, generated token count, throughput, failures, workload hash and grounding/approval checks. Rejected output and fallback are counted as failures. Warmups are recorded separately.

Throughput includes network and prefill time; it is not decode-only GPU throughput. The managed API does not independently report GPU hardware or VRAM; these remain unknown. Client hardware must never be presented as remote inference hardware.

Compare two completed runs with identical workload and generation settings:

```bash
python scripts/benchmark_inference.py --compare artifacts/run_a.json artifacts/run_b.json --output artifacts/comparison.json
```

No managed-service performance benchmark is claimed here. The released demo contains a genuine AMD evidence hearing, which is not a multi-case latency benchmark. Historical local-provider measurements belong to the earlier prototype and remain in Git history, not the current submission claims.
