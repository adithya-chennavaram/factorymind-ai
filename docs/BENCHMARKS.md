# Inference benchmarks

**AMD measurements: pending account access.** There is no AMD result, latency, token rate or speedup in this report.

## Measured local Ollama baseline

Recorded 2026-10-12T22:08:26.058678+00:00; run `81e697fe35ee4d568502165fa1406746`. Ollama 0.32.7, server-returned model `llama3.2`, 3.2B, quantization Q4_K_M. Host inventory: Apple M4, 10 GPU cores (system_profiler); per-request GPU execution and dedicated VRAM were not measured. This is not AMD inference.

Model digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.

One warmup took 4505.911 ms and is excluded from summary statistics. It is a warmup observation, not a measured cold-model-load time.

| Metric | Observed result |
|---|---:|
| Measured requests | 18 |
| Successful validated investigations | 18 |
| Failures | 0 |
| Median E2E latency | 1182.247 ms |
| p95 E2E latency | 1648.098 ms |
| Generated tokens (server reported) | 735 |
| Aggregate generated tokens / E2E seconds | 31.257 |

| Cycle | Dataset label | Served classifier prediction | Requests | Median E2E ms | p95 E2E ms | Failures |
|---|---|---|---:|---:|---:|---:|
| 4 | Optimal switching | Optimal switching | 3 | 1156.925 | 1178.247 | 0 |
| 213 | Near failure | Near failure | 3 | 1202.104 | 1590.972 | 0 |
| 222 | Severe lag | Severe lag | 3 | 1177.385 | 1597.831 | 0 |
| 232 | Small lag | Small lag | 3 | 1226.773 | 1612.519 | 0 |
| 248 | Optimal switching | Near failure | 3 | 1161.231 | 1598.255 | 0 |
| 496 | Severe lag | Small lag | 3 | 1183.878 | 1597.630 | 0 |

## Protocol and limits

The same six held-out cases cover all four valve labels and two classifier mistakes. There are three sequential repeats at concurrency 1; case ordering rotates across repeats. A separate warmup is retained. Temperature is zero, max_tokens is 300, top_p is 1, and the response format is JSON. The benchmark hashes the exact evidence prompts and classifier, and records source hashes for the provider, investigation and benchmark code.

End-to-end timing starts immediately before prediction and ends after the LLM response, schema/interpretation validation and factual rendering. It includes network, prompt processing and generation, and excludes model asset loading, connection/auth preflight and file writing. Provider-call latency is also retained. Generated tokens count the model’s JSON selection, not the longer application-rendered explanation. Throughput is sum of generated tokens divided by sum of E2E times over successful requests with usage; it is **not decode-only tokens/s**. Missing usage stays null and is excluded from the throughput denominator. Failed/rejected/truncated responses remain failures even if generation produced tokens. The small sample does not establish sustained capacity or a stable tail-latency distribution.

Successful bounded output is a protocol/grounding result, not proof of useful maintenance advice. Facts are rendered from the case; no free-form diagnosis is accepted. Unit tests include unsupported diagnoses, invented measurements, unsafe actions, contradiction, truncation, missing usage, wrong model attribution, network/auth failures and absence of automatic approval. Human review is still required; no external effectiveness study has been run.

## Reproduce and compare

```bash
source .venv/bin/activate
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.2
export LLM_API_KEY=ollama
export LLM_DISABLE_THINKING=0
export LLM_TEMPERATURE=0
export LLM_MAX_TOKENS=300
export LLM_TOP_P=1
unset LLM_MODEL_ID LLM_MODEL_REVISION LLM_HARDWARE_EVIDENCE
python scripts/benchmark_inference.py --output artifacts/ollama_benchmark.json
# After approved AMD setup and switching environment per AMD_RUNBOOK.md:
python scripts/benchmark_inference.py --output artifacts/amd_benchmark.json
python scripts/benchmark_inference.py --compare artifacts/ollama_benchmark.json artifacts/amd_benchmark.json --output artifacts/inference_comparison.json
```

Comparison requires actual completed runs with identical workload/classifier hashes, case list, repeats, warmup count and generation settings. Per-case results and failures remain visible. Different model weights, quantization, tokenizers and runtimes confound the comparison: any measured difference is a system comparison, not an isolated AMD-versus-Apple hardware speedup. No AMD comparison can be generated until a real remote run exists.

Raw benchmark records and outputs: `artifacts/ollama_benchmark.json` (local/ignored). The AMD profile refuses to benchmark without recent operator GPU evidence matching the unique deployment alias. That association is not cryptographic attestation; runtime GPU utilization and server-side request logs still need human verification.

Final run mechanical quality checks: **18/18 passed all four checks** (structured acceptance, exact evidence/catalog rendering, approval required, no equipment control). Expert usefulness review remains pending. GPU memory during local inference is **not measured**; AMD host sampling and evidence attachment are prepared in `AMD_RUNBOOK.md`. Do not interpret unavailable memory as zero.
