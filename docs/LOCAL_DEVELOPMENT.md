# FactoryMind AI — Local Development

This document contains setup and reproducibility commands moved out of the judge-first README. The application and model implementation are unchanged.

## Environment and installation

Tested locally on Apple Silicon with Python 3.13. The lock file captures the verified dependency environment.

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The repository intentionally excludes the UCI archive, processed data, trained model, review databases, secrets and local media.

## Dataset, training and audit artifacts

```bash
python -m scripts.download_data
python -m factorymind.train
python -m scripts.evaluate_green_review
python -m scripts.evaluate_green_challenge
```

The commands download the official UCI files, validate cycle/sample alignment, produce the 30-feature table, train the documented Random Forest, evaluate random and blocked splits, and generate the Green-Light and simulated-gain artifacts under ignored `artifacts/` paths.

## Streamlit

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. Select a held-out cycle, inspect the raw sensor trace and prediction, request an investigation, then approve or reject an inspection review. The Model Trust Lab is available in the sidebar. A local run remains useful without an AMD key through the clearly labeled rule-based fallback.

For a fresh hosted filesystem, set `FACTORYMIND_AUTO_SETUP=1` in the hosting secret configuration. The bootstrap downloads public data, trains the model and generates required Trust Lab artifacts once. A filesystem lock prevents duplicate concurrent initialization. Hosted runtime files are ephemeral and are never committed.

## FastAPI

In a second terminal:

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
source .venv/bin/activate
python -m uvicorn factorymind.api:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/docs`. Available routes include `/health`, `/provider/health`, `/cycles`, `/predict/{cycle_id}`, `/sensors/{cycle_id}/{sensor}`, `/investigate` and `/reviews`.

## AMD Token Factory configuration

Follow [AMD_TOKEN_FACTORY_ACTIII.md](AMD_TOKEN_FACTORY_ACTIII.md). Load the team key privately in the same terminal that launches the app; never put it in `.env`, source files, screenshots, logs or chat. The application uses the documented HTTPS API, validates the returned model and bounded JSON schema, reports provider identity and keeps fallback output separate from AMD output.

Public demonstration requests are protected by an 8-second cooldown, an 8-request per-session limit and a 60-request process cap by default. These controls reduce accidental repetition but are process-local and reset when Streamlit restarts; they are not authentication.

## Tests and checks

```bash
python -m pytest -q
git diff --check
```

The current verified suite contains 106 passing tests and one upstream Starlette/httpx deprecation warning. Tests cover data integrity, feature exclusion, blocked purging, predictions, API validation, provider failures, bounded investigations, fallback attribution, Streamlit workflow, bootstrap cleanup, quota controls and SQLite evidence audits.

## Evidence and limitations

The served model is trained on the original random-split training cycles; the UI identifies held-out cases. Scores are uncalibrated. Blocked validation is retrospective and uses the same experimental rig. Sensor-gain stress tests are hypothetical. SQLite evidence checksums detect changes to stored JSON but do not make the database immutable or tamper-proof. No machinery is controlled and no physical diagnosis, safety guarantee, future failure prevention or factory savings claim is supported.
