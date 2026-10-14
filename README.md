# FactoryMind AI — valve-condition investigation

A working local prototype for the AMD ACT III Intelligent Industry project. An operator selects a recorded hydraulic test-rig cycle, sees a real classifier prediction and sensor traces, requests an Ollama-assisted investigation, and explicitly approves or rejects an inspection review. No actuator commands exist.

**Local verification: October 12, 2026.** Real UCI data downloaded and validated; model trained; FastAPI, Streamlit, local Ollama and SQLite review flow exercised. **October 13 code update:** AMD ACT III Token Factory managed-provider adapter and offline tests added; the owner has obtained the team key, but **live AMD inference has not yet been tested**. This project is separate from SentaDQ. Confirm event eligibility and AI-assistance disclosure with organizers.

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

## AMD Token Factory — ready for owner's authenticated live test

The ACT III organizer-issued team manual specifies the AMD-managed API base URL `https://developer.amd.com.cn/radeon/api/v1` (distinct from the `/tokenfactory` sign-in website). The owner has obtained a team key. A dedicated `amd_tokenfactory` provider supports the documented OpenAI-compatible chat, selected model ID, JSON output, a lightweight authenticated `/models` health check, secret-safe error handling and honest provider attribution. New mocked tests pass. This is a **managed model service**, not a self-hosted MI300X instance. No live AMD call has yet been made in this environment because the secret key stays on the owner's Mac.

See **[AMD Token Factory ACT III quickstart](docs/AMD_TOKEN_FACTORY_ACTIII.md)**. Historical `amd_vllm`/ROCm instructions are preserved for reference only, not required for this main-track project. The optional Green-Light Audit baseline is described in [docs/GREEN_LIGHT_BASELINE.md](docs/GREEN_LIGHT_BASELINE.md) and requires regenerating blocked predictions to add out-of-fold probabilities.

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

Prior owner checkpoint: **79 tests passed** (local Mac). New October 13 changes add the Token Factory provider and Green-Light baseline tests: **35 targeted offline tests passed** in this restricted environment. The entire suite could not be rerun here because the local sandbox lacks `openai` and `streamlit`, and package installation was unavailable. Run `pytest -q` on the owner's complete Mac environment before submission.

`pytest -q` covers extraction/label alignment, malformed data, split leakage/purge, prediction, API validation/readiness, provider timeouts, bounded response validation, SQLite decisions and the Streamlit form flow. Unit fixtures are synthetic and never reported as real evaluation. The UI integration test uses real local artifacts and skips on a fresh checkout until training completes. `requirements-lock.txt` pins the environment; `requirements.txt` lists allowed dependency ranges.

Raw files, trained models, environment secrets and local reviews are ignored. The final checkpoint initializes a local-only Git repository; see `docs/HANDOFF.md` for the branch and checkpoint reference. No push or publication is authorized. Keep these files out of Git. Never load an untrusted joblib artifact.

## Data attribution, licenses and disclosure

Data: Helwig, N., Pignanelli, E., & Schütze, A. (2015), *Condition monitoring of hydraulic systems*, UCI Machine Learning Repository, [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21), **CC BY 4.0**. Source: [official UCI dataset 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems). We derive per-cycle statistical features from five sensors. This is an experimental hydraulic rig, not a real factory telemetry feed.

Project code uses the MIT license in `LICENSE`; the dataset and downloaded model weights have separate licenses. Review all third-party redistribution terms before publishing.

The starter was generated with ChatGPT assistance. Codex expanded the implementation, installed dependencies, downloaded public data, trained/evaluated the model, and tested the local applications. Owner review and organizer confirmation of AI coding-assistance rules are pending. No AMD hardware use, contest score, eligibility, cost savings or deployment is claimed without evidence.

### AMD ACT III Token Factory (managed API)

The official ACT III team API is **not** the historical self-hosted vLLM deployment. See [`docs/AMD_TOKEN_FACTORY_ACTIII.md`](docs/AMD_TOKEN_FACTORY_ACTIII.md) for secure environment setup, model selection and the first live probe. The code now accepts `LLM_PROVIDER=amd_tokenfactory` with documented HTTPS API URL and a privately exported `rc-` team key; live inference must still be verified by the owner.
