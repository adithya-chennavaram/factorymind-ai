# FactoryMind AI — Model Trust Lab

FactoryMind AI — Model Trust Lab audits industrial model predictions before an operator relies on an “all clear.” It combines blocked-validation analysis, false-green review, hypothetical sensor-gain stress tests, an AMD Token Factory evidence hearing, and a human-approved Decision Passport. The local prototype uses recorded UCI hydraulic test-rig cycles; it does not control equipment or claim real-factory safety.

**Verified locally:** UCI data, classifier evaluation, FastAPI, Streamlit and SQLite review flow. The recorded October 13 demo also exercised AMD Token Factory (`self-dploy/DeepSeek-V4-Flash`), bounded evidence selection and inspection approval #3. The narrated demo was completed October 17. This is managed-service inference evidence, not independent attestation of the underlying GPU. The project is independent of other projects; confirm event eligibility and AI-assistance disclosure with organizers.

## Hackathon demonstration

**Submission title:** FactoryMind AI — Model Trust Lab · **Track:** Intelligent Industry

- [Watch/download the narrated demo (2:55)](https://github.com/adithya-chennavaram/factorymind-ai/releases/download/act-iii-submission/FactoryMind_ACTIII_demo.mp4)
- [Pitch slides (PDF)](https://github.com/adithya-chennavaram/factorymind-ai/releases/download/act-iii-submission/FactoryMind_ACTIII_Pitch.pdf) · [Editable PowerPoint](https://github.com/adithya-chennavaram/factorymind-ai/releases/download/act-iii-submission/FactoryMind_ACTIII_Pitch.pptx)
- [Submission cover](https://github.com/adithya-chennavaram/factorymind-ai/releases/download/act-iii-submission/FactoryMind_Cover.png)

The recording shows blocked-validation errors, simulated sensor-gain checks, an AMD evidence hearing, and a human-approved Decision Passport. Your own API key is required for live AMD inference. There is currently **no hosted interactive deployment**; reproduce the application locally with the commands below. Release assets contain the final presentation and demo, while raw data, trained models, credentials and local audit databases remain excluded.

## Run locally

Tested on Apple Silicon with Python 3.13. The lock file captures this verified environment.

```bash
cd factorymind-ai
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
cd factorymind-ai
source .venv/bin/activate
uvicorn factorymind.api:app --host 127.0.0.1 --port 8001
```

Open [API documentation](http://127.0.0.1:8001/docs). Both services are local development applications. Keep them bound to loopback; authentication, authorization and production audit protection are not implemented. Restart both after retraining or changing provider environment variables, because model resources are cached.

## AMD Token Factory investigation

The application uses the ACT III managed AMD Token Factory API. Follow the [secure configuration guide](docs/AMD_TOKEN_FACTORY_ACTIII.md) to export your private key and launch the app in the same terminal. `.env.example` documents supported variables; it is not automatically loaded. No GPU instance or container is required.

Without a valid key, predictions and sensor charts still work. Investigation returns a clearly labeled **rule-based fallback**, never an AMD-generated answer. The LLM selects observations and safe review checks from a bounded catalog; unknown sensors, unsupported claims, malformed output and mismatched models are rejected. Dataset labels are excluded from prompts, and human approval is always explicit.

```bash
python scripts/probe_llm.py --cycle 213
```

This command performs authenticated model discovery and one real investigation using team quota. Local probe outputs remain ignored. [Benchmark instructions](docs/BENCHMARKS.md) describe optional measured inference runs; no new inference benchmark was run during cleanup.

## Reproduce the Model Trust Lab demo

The main dashboard and the Model Trust Lab use different model evaluations. After downloading the UCI data and running `python -m factorymind.train` above, generate the blocked-review and simulated-gain artifacts locally:

```bash
python -m scripts.evaluate_green_review
python -m scripts.evaluate_green_challenge
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **Model Trust Lab** in the sidebar. Compare cycles **2053** (stable but wrong) and **507** (26/50 simulated class switches) at a 20% review budget. Both are retrospective examples. Configure the AMD provider using the Token Factory guide before requesting an evidence hearing. After an accepted response, explicitly approve or reject an inspection review and download its Decision Passport. Review IDs depend on the local database; no pre-existing decision is required. Generated datasets, model artifacts and review databases remain local and ignored.

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
- [AMD Token Factory setup](docs/AMD_TOKEN_FACTORY_ACTIII.md)

## API

`GET /provider/health` separately checks live provider health, model availability. `GET /health` reports model readiness and dataset count; missing assets return 503. `GET /cycles?limit=20&offset=0` paginates cycle IDs. `GET /predict/{cycle_id}` returns prediction, factual evidence, annotation and latency. `GET /sensors/{cycle_id}/{sensor}` returns a trace with units, time axis and explicit display stride. `POST /investigate` accepts `{"cycle_id":213}`. `POST /reviews` accepts `{"cycle_id":213,"decision":"approve_inspection","notes":"Operator note"}` or `reject_inspection`. `GET /reviews/{cycle_id}` reads recent local decisions.

## Tests and reproducibility

Latest local suite run, October 17, 2026: **102 tests passed**, with one upstream Starlette/httpx deprecation warning. Legacy host-deployment tests were removed with that unused feature; active provider, grounding, approval and UI tests remain. Run `python -m pytest -q` to reproduce.

`pytest -q` covers extraction/label alignment, malformed data, split leakage/purge, prediction, API validation/readiness, provider timeouts, bounded response validation, SQLite decisions and the Streamlit form flow. Unit fixtures are synthetic and never reported as real evaluation. The UI integration test uses real local artifacts and skips on a fresh checkout until training completes. `requirements-lock.txt` pins the environment; `requirements.txt` lists allowed dependency ranges.

Raw files, trained models, environment secrets, local reviews, agent/session notes and video production outputs are excluded from Git. The repository contains application source, tests, configuration templates and reproducibility documentation. Never load an untrusted joblib artifact.

## Data attribution, licenses and disclosure

Data: Helwig, N., Pignanelli, E., & Schütze, A. (2015), *Condition monitoring of hydraulic systems*, UCI Machine Learning Repository, [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21), **CC BY 4.0**. Source: [official UCI dataset 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems). We derive per-cycle statistical features from five sensors. This is an experimental hydraulic rig, not a real factory telemetry feed.

Project code uses the MIT license in `LICENSE`; the dataset and downloaded model weights have separate licenses. Review all third-party redistribution terms before publishing.

The starter was generated with ChatGPT assistance. Codex expanded the implementation, installed dependencies, downloaded public data, trained/evaluated the model, and tested the local applications. Owner review and organizer confirmation of AI coding-assistance rules are pending. No AMD hardware use, contest score, eligibility, cost savings or deployment is claimed without evidence.
