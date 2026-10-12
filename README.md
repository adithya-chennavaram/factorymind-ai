# FactoryMind AI — valve-condition investigation

A working local prototype for the AMD ACT III Intelligent Industry project. An operator selects a recorded hydraulic test-rig cycle, sees a real classifier prediction and sensor traces, requests an Ollama-assisted investigation, and explicitly approves or rejects an inspection review. No actuator commands exist.

**Local verification: October 12, 2026.** Real UCI data downloaded and validated; model trained; FastAPI, Streamlit, local Ollama and SQLite review flow exercised. **AMD GPU inference has not run.** This project is separate from SentaDQ. Contest eligibility and AI-assistance permission remain subject to organizer confirmation.

## Run locally

Tested on Apple Silicon with Python 3.13. The lock file captures this verified environment.

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/download_data.py
python -m factorymind.train
pytest -q
streamlit run app.py
```

Open [the operator demo](http://127.0.0.1:8501). In another terminal:

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
source .venv/bin/activate
uvicorn factorymind.api:app --host 127.0.0.1 --port 8001
```

Open [API documentation](http://127.0.0.1:8001/docs). Both services are local development applications. Keep them bound to loopback; authentication, authorization and production audit protection are not implemented. Restart both after retraining or changing provider environment variables, because model resources are cached.

## Local AI investigation

Ollama and `llama3.2` were already installed on this Mac and successfully exercised. On a fresh machine, install Ollama from its official distribution, then:

```bash
ollama pull llama3.2
ollama serve  # only if the service is not already running
```

Default provider settings are `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2`, `LLM_API_KEY=ollama`, and `LLM_TIMEOUT_SECONDS=60`. Generation settings are `LLM_TEMPERATURE=0`, `LLM_MAX_TOKENS=300`, and `LLM_TOP_P=1`; validated ranges are 0–2, 64–1024 and (0,1] respectively. Settings are read from the process environment; `.env.example` is a reference and is not automatically loaded.

The LLM selects sensor observations and safe review checks in a strict JSON schema. The application rejects unknown sensors, extra claims, unsafe actions, contradictory interpretations, malformed output and truncated responses. Rendered measurements come from the recorded case, and the dataset label is excluded from the LLM prompt. This is **bounded AI selection with factual template rendering**, not free-form causal diagnosis or validated feature attribution. Errors and timeouts produce a clearly labeled rule-based fallback.

```bash
python scripts/probe_llm.py
python scripts/smoke_api.py
```

The probe saves authenticated health/model checks and a real investigation to `artifacts/provider_probe.json`. `--require-amd-evidence` fails on a local Ollama connection and on missing/stale remote hardware evidence. The API smoke test requires the API to be running and adds clearly marked **automated test** approval/rejection records to the local audit. It writes `artifacts/local_smoke.json`.

## AMD Token Factory — integration pending

Owner reports ACT III organizer confirmation that the Intelligent Industry track receives an API key for AMD-hosted models through [Token Factory](https://notebooks.amd.com/tokenfactory). Self-provisioned MI300X, ROCm installation and vLLM deployment are **not required for the main track**.

**Access blocker:** the registered email is not recognized for API key distribution; the owner has escalated this to organizers. No actual key, endpoint, model ID or hosted inference has been verified.

Working now: local Ollama, structured investigation validation, explicit provider/fallback attribution, timeout handling and the measured local benchmark. Existing `amd_vllm` code and mocked tests target a self-hosted server. **A Token Factory provider and its mocked integration tests are not implemented.** Do not assume the existing `/health`, negative-auth checks, model aliases or GPU-evidence gate match the hosted service. The portal URL is not an inferred API base URL.

Next session should use the issued service documentation to add a small hosted-provider profile, adapt readiness/benchmark provenance checks, test it with mocks, and then verify authorized real requests. Keep missing hardware telemetry explicitly unavailable; never attribute local/fallback output to AMD. Historical deployment tooling is retained but inactive; do not provision anything.

See [handoff and exact resume steps](docs/HANDOFF.md), [measured Ollama benchmark](docs/BENCHMARKS.md), and [blocked-error analysis](docs/BLOCKED_ERROR_ANALYSIS.md).

## Actual model evaluation

2,205 cycles; 30 features from PS1, PS2, FS1, TS1 and VS1. Profile labels, stability and cycle IDs never enter the feature list. The training CLI checks 6,000 samples per pressure cycle, 600 flow samples, 60 temperature/vibration samples, row alignment and finite values. SHA-256 download hashes are retained in `data/raw/manifest.json`; these are local provenance hashes, not publisher-signed checksums.

| Evaluation | Evaluated cycles | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Random stratified holdout | 552 | 98.01% | 97.89% | 0.9745 |
| Five contiguous folds, 10-cycle purge | 2,205 | 82.86% | 75.69% | 0.7877 |
| Majority-class holdout baseline | 552 | 51.09% | 25.00% | 0.1691 |

Random splitting is optimistic because neighboring rig cycles correlate. The blocked evaluation pools out-of-fold predictions; each fold holds out one contiguous block and excludes ten adjacent cycles on either side from training. It can train on both earlier and later cycles and is **not a future-forecasting evaluation or independent-rig validation**. Neither result establishes operational plant performance. In blocked evaluation, small-lag recall is only 48.89%.

The served random forest remains trained on the original 1,653 training cycles so that held-out demo cases are genuinely unseen by that model. The UI labels training versus held-out cases. Scores are uncalibrated and are not failure probabilities. Full matrices, per-class results, fold results, split IDs, model hash and versions are in `artifacts/metrics.json`; a readable snapshot is in [docs/EVALUATION.md](docs/EVALUATION.md).

## Demonstration and audit

Use **cycle 4** (correct optimal prediction), **213** (correct near-failure class) and **248** (incorrect near-failure prediction; actual label optimal). The sidebar exposes held-out non-optimal, optimal and error collections. Compare recorded sensor traces against an optimal-label reference, investigate, and use the explicit approval or rejection button. Decisions store a real review timestamp, note and evidence snapshot in `data/reviews.sqlite3`; timestamps describe review events, not invented sensor acquisition times. UI reviews retain the investigation; API reviews retain the prediction at decision time.

The database is a local demonstration log, not a tamper-evident or authenticated audit system. Automated smoke records are marked in their notes. Nothing creates a physical work order or operates machinery.

- [Demo script](docs/DEMO_SCRIPT.md)
- [Judging evidence and blockers](docs/JUDGING_EVIDENCE.md)
- [AMD ROCm / vLLM runbook](docs/AMD_RUNBOOK.md)

## API

`GET /provider/health` separately checks live provider health, model availability and (for AMD vLLM) rejection of missing/incorrect keys. `GET /health` reports model readiness and dataset count; missing assets return 503. `GET /cycles?limit=20&offset=0` paginates cycle IDs. `GET /predict/{cycle_id}` returns prediction, factual evidence, annotation and latency. `GET /sensors/{cycle_id}/{sensor}` returns a trace with units, time axis and explicit display stride. `POST /investigate` accepts `{"cycle_id":213}`. `POST /reviews` accepts `{"cycle_id":213,"decision":"approve_inspection","notes":"Operator note"}` or `reject_inspection`. `GET /reviews/{cycle_id}` reads recent local decisions.

## Tests and reproducibility

Final checkpoint: **79 tests passed**, one upstream Starlette/httpx deprecation warning (2.32 seconds). Tests were run once; no new dependencies or provider implementation were added.

`pytest -q` covers extraction/label alignment, malformed data, split leakage/purge, prediction, API validation/readiness, provider timeouts, bounded response validation, SQLite decisions and the Streamlit form flow. Unit fixtures are synthetic and never reported as real evaluation. The UI integration test uses real local artifacts and skips on a fresh checkout until training completes. `requirements-lock.txt` pins the environment; `requirements.txt` lists allowed dependency ranges.

Raw files, trained models, environment secrets and local reviews are ignored. The final checkpoint initializes a local-only Git repository; see `docs/HANDOFF.md` for the branch and checkpoint reference. No push or publication is authorized. Keep these files out of Git. Never load an untrusted joblib artifact.

## Data attribution, licenses and disclosure

Data: Helwig, N., Pignanelli, E., & Schütze, A. (2015), *Condition monitoring of hydraulic systems*, UCI Machine Learning Repository, [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21), **CC BY 4.0**. Source: [official UCI dataset 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems). We derive per-cycle statistical features from five sensors. This is an experimental hydraulic rig, not a real factory telemetry feed.

Project code uses the MIT license in `LICENSE`; the dataset and downloaded model weights have separate licenses. Review all third-party redistribution terms before publishing.

The starter was generated with ChatGPT assistance. Codex expanded the implementation, installed dependencies, downloaded public data, trained/evaluated the model, and tested the local applications. Owner review and organizer confirmation of AI coding-assistance rules are pending. No AMD hardware use, contest score, eligibility, cost savings or deployment is claimed without evidence.
