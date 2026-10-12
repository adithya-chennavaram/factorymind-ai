# Verification evidence and remaining milestones

> **Current owner update — Token Factory:** The main Intelligent Industry track uses organizer-issued AMD-hosted model API access at https://notebooks.amd.com/tokenfactory. The registered email is not recognized; organizer escalation is pending. Self-hosted GPU deployment is superseded for this track. Retain the instructions below as historical/optional preparation only; do not provision or deploy. Token Factory integration is not yet implemented. Follow `docs/HANDOFF.md` (or `HANDOFF.md` from this directory) for the current next steps.


## AMD readiness update — October 12, 2026

Owner confirmed that AMD Developer Cloud access, GPU/VRAM, ROCm version and container restrictions are still pending. No hardware selection or cloud action was made. Added explicit provider profiles, returned-model attribution, typed errors/timeouts, authenticated health checks, operator evidence linkage, a HIP/GPU collector, a VRAM-dependent offline model planner, a private Compose template and repeatable inference benchmarks. The missing-hardware planner and local-connection AMD gate were exercised and correctly refused to claim AMD readiness.

The final local baseline measured 18/18 successful investigations, zero failures, median E2E 1,182.247 ms, p95 1,648.098 ms, 735 generated tokens and 31.257 generated tokens per E2E second. See `BENCHMARKS.md` for exact method, model digest, quantization and limitations. Raw response records are in `artifacts/ollama_benchmark.json`. No AMD comparison exists yet. `BLOCKED_ERROR_ANALYSIS.md` quantifies confusion directions, block concentration and descriptive annotation associations.

Container YAML syntax was checked with Ruby's YAML parser. Docker is not installed locally, so actual Compose validation/startup, ROCm drivers, model load, authenticated AMD requests and GPU utilization remain unexecuted. UI wording distinguishes local Ollama, vLLM with unverified GPU, vLLM with matching operator evidence, and rule-based fallback. Client code never asserts physical hardware attestation.

Final live API check: `/health` reported 2,205 cycles; `/provider/health` was ready; cycle 213 returned `served_by=ollama` in 1,519.15 ms, with no AMD attribution and no implicit review write. Saved to `artifacts/provider_api_smoke.json`. Both services were restarted with final code and remain bound to loopback (8501 and 8001).

Current preparation suite: **79 tests passed**; one upstream Starlette/httpx deprecation warning. Includes mocked vLLM completion through the real SDK, configuration/security failures, factual grounding, human approval, single-GPU admission, lifecycle cleanup and telemetry identity/time-window checks. All 18 live Ollama benchmark responses passed the four mechanical quality checks. AMD host execution remains pending.

## Earlier local prototype verification — October 12, 2026

- Official UCI archive downloaded over verified TLS; six required files extracted. Archive/file hashes and download timestamp: `data/raw/manifest.json`.
- Training validated 2,205 cycles, 30 sensor features, exact sample counts, finite values and row alignment. Label selection uses the second profile column. Full evaluation: `artifacts/metrics.json` and `docs/EVALUATION.md`.
- Random holdout: 1,653 training cycles, 552 test cycles, 541 correct and 11 incorrect predictions; accuracy 98.0072%, macro F1 0.974515.
- Five contiguous folds with ten-cycle boundary purge: pooled accuracy 82.8571%, macro F1 0.787744. Small-lag recall 48.8889%. This is a meaningful limitation, not a hidden result.
- Real API requests: `/health`, `/predict`, `/sensors`, `/investigate`, `/reviews` and review retrieval. A PS1 response confirmed 6,000 original samples and 600 display samples.
- Live API investigations through local Ollama `llama3.2`: cycle 4, 2,130.08 ms; cycle 213, 1,151.09 ms; cycle 248, 1,607.91 ms. All three returned `provider_mode=llm`. These are individual observed requests, not a latency benchmark or AMD results. Evidence: `artifacts/local_smoke.json`.
- Independent provider probe listed the available model and completed cycle 213 in 1,221.53 ms: `artifacts/provider_probe.json`. Initial direct cycle-1 invocation took 4,839.92 ms: `artifacts/local_investigation.json`.
- Streamlit loaded in a browser with real traces, reference comparison, score and model identity; a live AI investigation was displayed. The original radio decision control failed a browser interaction check and was replaced by explicit approve/reject submit buttons. Both revised buttons were exercised in the browser and verified in SQLite: review ID 5 is `reject_inspection`; ID 6 is `approve_inspection`.
- Automated suite: **26 passed**, one upstream Starlette/httpx deprecation warning, 1.78 seconds on the final recorded suite run. Includes real-artifact Streamlit investigation fallback, both decision buttons, persisted evidence and stale-case hiding. Other tests use explicitly synthetic fixtures and are not evaluation results.
- API and browser smoke records are clearly marked `AUTOMATED ... SMOKE TEST` in SQLite notes. These do not represent real maintenance decisions; development records are retained rather than silently deleted.

## Evidence locations

| File | Meaning |
|---|---|
| `data/raw/manifest.json` | Official URL, actual download timestamp and local integrity hashes |
| `artifacts/metrics.json` | Complete per-class metrics, both confusion matrices, split IDs, fold reports, versions and model hash |
| `artifacts/predictions.csv` | Served-model predictions for all cycles, marked training/held-out |
| `artifacts/blocked_predictions.csv` | Out-of-fold blocked predictions |
| `artifacts/local_smoke.json` | Real API investigations, latencies, readiness and test audit retrieval |
| `artifacts/provider_probe.json` | Current provider health/auth/model checks and validated investigation; no GPU claim |
| `data/reviews.sqlite3` | Decision, note, review timestamp and snapshot of evidence |
| `requirements-lock.txt` | Installed dependency versions |

The raw data, models, local traces and audit database remain local and ignored by Git. The workspace began as a source folder without `.git`; no commit, remote repository or upload was created.

## Remaining blockers and limits

**AMD execution:** not performed. Need owner-approved cloud access, verified credits/rates and a spending cap. Follow `AMD_RUNBOOK.md`, collect actual server GPU evidence and demonstrate the product using the AMD endpoint. A model name or client response does not prove hardware provenance.

**Contest rules:** owner is awaiting organizer confirmation of AI coding-assistance permissions and final competition requirements. Keep the disclosure intact. No claims of score, ranking or prize eligibility are verified here.

**Submission:** public repository, hosted deployment, video, slides and actual submission are later milestones requiring owner approval and review. No paid resources, accounts or public deployment were launched.

**Scientific/product limits:** experimental rig only; no external-plant validation, probability calibration, causal explainability, real maintenance history or measured business benefits. The LLM chooses from a bounded catalog; it does not generate an independent diagnosis. UI reference comparison is descriptive and may differ in other rig conditions. Scores must not be interpreted as failure probability. Local audit has no operator authentication or tamper protection. Both local services must remain bound to loopback.

## Owner's next action

Review the local demo, then provide approved AMD GPU access/credit confirmation and the allowed budget. Confirm organizer guidance before publication or submission. No action is required to reproduce the already working local prototype beyond running the README commands if the development servers have stopped.
