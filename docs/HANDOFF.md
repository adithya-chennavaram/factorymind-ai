# FactoryMind AI — final checkpoint

Checkpoint date: 2026-10-12. Track: AMD ACT III Intelligent Industry. Team: FactoryMind AI. Independent of SentaDQ. Stop after this checkpoint; no scheduled continuation.

## Git checkpoint

Branch: `main`. Implementation checkpoint commit: `1243d1868df57234f5977ae6b8a5a11373a51312`.
The final documentation commit records the implementation commit above. Obtain the current documentation-inclusive HEAD with `git rev-parse HEAD`; a committed document cannot contain its own eventual Git hash. No remote, push or publication is authorized.

## Implemented features and modules

- `factorymind/data.py`, `train.py`, `scripts/download_data.py`: real UCI hydraulic dataset extraction, strict cycle/sample/label alignment, 30 sensor features, random-forest training and random/blocked evaluation. Labels, stability and cycle IDs excluded from predictors.
- `factorymind/core.py`: real predictions, factual sensor evidence, bounded JSON LLM selection, grounded rendering and transparent rule-based fallback. Classifier output is a hypothesis, not a physical diagnosis.
- `factorymind/providers.py`: local Ollama and self-hosted `amd_vllm` profiles, validated settings, explicit returned-model/provider attribution, authentication checks, timeouts and redacted failures. No silent fallback to another LLM.
- `factorymind/api.py`: FastAPI readiness, provider health, cycles, predictions, sensor traces, investigations and reviews.
- `app.py`, `.streamlit/config.toml`: loopback Streamlit demo, raw signals, correct/error case collections, explicit approve/reject buttons and provider/latency display.
- `factorymind/reviews.py`: SQLite decisions and evidence snapshots; no actuator control or automatic approval.
- `factorymind/benchmark.py`, `scripts/benchmark_inference.py`, `probe_llm.py`: repeatable held-out investigations, latency, generated tokens, failure counts, mechanical grounding/approval checks and comparison support.
- `scripts/analyze_errors.py`, `docs/EVALUATION.md`, `BLOCKED_ERROR_ANALYSIS.md`, `BENCHMARKS.md`: honest measured evaluation and limitations.
- `factorymind/deployment.py`, `scripts/prepare_amd.py`, `collect_amd_hardware.py`, `amd_host.py`, `deploy/compose.amd.yaml`: historical self-hosted preparation only. Retained but unnecessary for the updated main-track route; no Docker/ROCm run was performed.
- `tests/`: model/data integrity, API, UI, review audit, provider auth/errors, mocked vLLM completion, grounding, deployment planning and telemetry checks.

## Exact local commands

Existing local environment and assets are present. Use separate terminals for the two servers; avoid starting duplicates if ports are occupied.

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
source .venv/bin/activate
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.2
export LLM_API_KEY=ollama
export LLM_TIMEOUT_SECONDS=60
export LLM_TEMPERATURE=0
export LLM_MAX_TOKENS=300
export LLM_TOP_P=1
export LLM_DISABLE_THINKING=0
unset LLM_MODEL_ID LLM_MODEL_REVISION LLM_HARDWARE_EVIDENCE
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

FastAPI, in another terminal with the same environment (defaults also select local Ollama):

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
source .venv/bin/activate
python -m uvicorn factorymind.api:app --host 127.0.0.1 --port 8001
```

Tests:

```bash
cd /Users/adithyachennavaram/Downloads/factorymind-ai
.venv/bin/python -m pytest -q
```

URLs: http://127.0.0.1:8501/ and http://127.0.0.1:8001/docs. Ollama must already be running with `llama3.2`; use `ollama serve` only if stopped and `ollama pull llama3.2` only if missing. Restart apps after changing provider settings.

For a fresh checkout only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/download_data.py
python -m factorymind.train
```

Python 3.13 / Apple Silicon was verified. Dependencies: numpy, pandas, scikit-learn, joblib, Streamlit, FastAPI, uvicorn, OpenAI-compatible Python client, httpx, certifi and pytest. Exact installed versions are in `requirements-lock.txt`; ranges in `requirements.txt`. `.env.example` documents settings but is not automatically loaded. Actual keys belong only in environment/secret storage, never Git, logs or chat.

## Latest verified results

Final checkpoint ran the existing suite **once: 79 passed in 2.32 seconds**, one upstream Starlette/httpx deprecation warning. No failures or skips. This includes mocked self-hosted vLLM integration, not Token Factory integration. No retraining or live benchmark rerun during this checkpoint.

Previously measured real UCI evaluation: 2,205 cycles, 30 features across PS1, PS2, FS1, TS1, VS1. Original served model trained on 1,653 cycles; 552 held out.

| Evaluation | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Random stratified holdout | 98.01% | 97.89% | 0.9745 |
| Five contiguous folds, 10-cycle purge | 82.86% | 75.69% | 0.7877 |

Random holdout: 541 correct, 11 errors. Blocked: 378 errors; 260 non-optimal cycles predicted optimal; small-lag recall 48.89%. Correlated rig cycles make random splitting optimistic. Neither evaluation establishes plant performance or causal failure diagnosis; scores are uncalibrated. Full retained records are in local ignored `artifacts/metrics.json`, `predictions.csv`, `blocked_predictions.csv`; readable results are committed under `docs/`.

Previously measured Ollama baseline: 18/18 accepted investigations, no failures; median E2E 1,182.247 ms, p95 1,648.098 ms, 735 generated tokens, 31.257 generated tokens/E2E second. All 18 passed mechanical grounding/approval checks. Expert usefulness review remains pending. Model `llama3.2`, Ollama 0.32.7, Apple M4 inventory; no per-request GPU execution claim. See `docs/BENCHMARKS.md` and ignored `artifacts/ollama_benchmark.json`.

Demo cases: 4 correct optimal; 213 correct near-failure class; 248 incorrect near-failure prediction despite optimal dataset label. No physical failure is established. Previous live API check verified 2,205 cycles, Ollama attribution and no implicit review write.

## Current AMD access blocker and next connection steps

The owner reports organizer confirmation that teams receive an API key for AMD-hosted models through https://notebooks.amd.com/tokenfactory. **The registered email is not recognized for API key distribution; the owner has escalated to organizers.** Do not provision MI300X, install ROCm or deploy vLLM for the main Intelligent Industry track. Previous $100-credit/cloud plans are historical, not current requirements or spending authorization.

**Token Factory provider is not implemented.** Only `ollama` and `amd_vllm` are accepted. No Token Factory endpoint, model, auth contract, quota, GPU details or real response has been verified. Existing self-hosted `/health`, model aliases and local GPU-evidence gates must not be assumed compatible with the hosted API.

Once access is granted:

1. Retrieve the key privately and obtain official service documentation: actual API base URL, available model IDs, auth scheme, request/response format, JSON-output support, token/timeout limits, usage/quota terms and AMD-hosting provenance. The portal URL is not an API base URL. Do not assume Qwen or a particular GPU is available.
2. Add a small explicit Token Factory provider profile in `providers.py` using the confirmed contract. Reuse the client only if OpenAI compatibility is verified; adapt health/model/auth checks. Update `core.py` attribution and benchmark admission rules so hosted AMD responses do not require self-hosted HIP inventory. Keep unreported hardware/memory unknown, and distinguish organizer-provided hosting provenance from physical attestation. Do not relabel it as self-hosted vLLM.
3. Add mocked success/auth/timeout/malformed-response tests plus grounding, approval and actual-provider attribution checks. No real credentials in fixtures. Preserve Ollama behavior.
4. Export the documented base URL, model and private key under the supported environment variables after the profile exists. Enter the key via a hidden prompt/secret manager; restart both apps. Run one authorized real connection/investigation, verify returned provider/model and sanitized evidence, then run the same six-case benchmark against the retained Ollama baseline. Respect confirmed quotas; no paid requests without approval.
5. Demonstrate healthy, degraded and misclassified cases with human review. Store non-secret request/response evidence; never describe fallback/local output as AMD-generated.

## Known issues and unfinished work

- Token Factory account authorization and implementation are pending; no AMD inference or AMD-versus-Ollama comparison exists.
- Mocked self-hosted tests are not proof of hosted-service compatibility. Historical Docker template syntax was checked, but Docker/ROCm execution was not tested.
- Upstream Starlette/httpx warning remains; no dependency migration is needed for this checkpoint.
- Mechanical grounding is not expert maintenance usefulness. No causal explanation, calibration or independent-plant validation.
- Local API/UI and SQLite audit are not production-authenticated or tamper-evident. Bind to loopback. Automated audit entries are clearly labeled test records.
- Data, model artifacts and virtual environment are intentionally not in Git; retain this local workspace or reproduce them with download/train commands. Do not load untrusted joblib files.

## Git hygiene and submission tasks

Stage only reviewed source, tests, docs, dependency manifests, configuration templates and empty `.gitkeep` placeholders. Exclude `.env`, real keys, raw/processed data, model artifacts, SQLite reviews, logs, caches, virtual environments, host evidence and temporary files. No remote is configured and nothing is pushed. The staged-source review found no real credentials, large data/model files or temporary files; the only secret-pattern match was the deliberate `https://user:secret@gpu.example/v1` rejection fixture in `tests/test_providers.py`.

Remaining submission work: resolve Token Factory access; implement and verify real hosted AMD inference; review investigation usefulness; confirm organizer eligibility/AI-assistance disclosure rules; owner-approved repository publication and any required hosted demo; demo recording, pitch slides, cover image, dataset CC BY 4.0 attribution and separate code/model licenses; owner review and actual submission. No score, eligibility, benefits or AMD runtime result is invented.

Next session priorities: (1) resolve the email/API authorization; (2) implement/test the documented hosted-provider contract and benchmark real AMD responses; (3) prepare and review the submission materials with explicit publication approval.
