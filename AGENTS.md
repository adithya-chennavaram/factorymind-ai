# FactoryMind AI — Instructions for autonomous coding agents

> **Current owner update — Token Factory:** The main Intelligent Industry track uses organizer-issued AMD-hosted model API access at https://notebooks.amd.com/tokenfactory. The registered email is not recognized; organizer escalation is pending. Self-hosted GPU deployment is superseded for this track. Retain the instructions below as historical/optional preparation only; do not provision or deploy. Token Factory integration is not yet implemented. Follow `docs/HANDOFF.md` (or `HANDOFF.md` from this directory) for the current next steps.


You are the lead implementation agent on a **time-boxed AMD ACT III hackathon** project.
Read this file, README.md, docs/AMD_RUNBOOK.md, and docs/CODEX_MASTER_PROMPT.md before editing.

## User intent and constraints
- Owner wants **minimal manual work**. Build, run, test, fix, and document autonomously where permissions allow.
- Goal: a **working local prototype within 24 hours**, then meaningful AMD GPU inference and polished contest submission.
- Project independent from SentaDQ. **Do not reuse or rename SentaDQ.**
- User is on an Apple Silicon Mac M4. ROCm runs on an AMD GPU **cloud instance**, not this Mac.
- AI coding-assistance rules: the owner is checking confirmation with organizers. Keep an accurate disclosure; do not misrepresent development provenance.
- Do not create accounts, request billing authorization, spend money, publish repositories, submit entries, or upload private data without explicit owner approval.
- Avoid any claimed hackathon score, prize eligibility, credits, or leaderboard ranking not verified by sources.

## AMD deployment policy (owner specification)
- Requested $100 promotional credits are pending, not available funds. Out-of-pocket cap is $0 without explicit approval. Verify actual pricing and receive allocation/credit-use authorization before provisioning.
- Desired allocation only: exactly one MI300X (gfx942), advertised 192 GB VRAM, 20 vCPU and 240 GB RAM, compatible Ubuntu. Actual hardware remains unknown. Never select eight GPUs or mask an eight-GPU allocation as one.
- Prefer Qwen/Qwen2.5-7B-Instruct; use a smaller compatible model only when observed memory or recorded runtime failures prevent reliable operation, and document why.
- Use an official stable vllm/vllm-openai-rocm release compatible with observed hardware/driver, then pin its tested digest and the model commit. Never replace the cloud-provided driver speculatively.
- Keep endpoint private and authenticated. Record provider identity, GPU evidence, actual token/latency/failure measurements and sampled VRAM. A fallback must never claim AMD generation.
- Stop containers/tunnels after testing; separately delete the approved instance and unnecessary storage through the provider and verify billing termination. No cloud deletion or spending occurs without the owner's authorization.

## Goal / user story
A maintenance operator chooses a test-rig cycle, sees a **real model prediction** of valve condition, sees **real sensor observations**, requests an LLM explanation, and approves/rejects an **inspection request**. No autonomous equipment control.

## Dataset and modeling integrity
- Source: UCI Condition monitoring of hydraulic systems, dataset ID 447, DOI 10.24432/C5CW21, CC BY 4.0.
- 2,205 cycles, 60 seconds/cycle, hydraulic experimental test rig. This is **not real factory plant telemetry**.
- `profile.txt` columns: cooler, valve, internal pump leakage, accumulator, stability. Valve column = second column. Codes: 100 optimal, 90 small lag, 80 severe lag, 73 near failure.
- Label columns and stability flag MUST NOT become model predictors. Cycle ID likewise excluded.
- IMPORTANT: Random stratified split is a baseline that can overstate generalization because neighboring rig cycles correlate. Test group/blocked split where feasible, report both evaluations transparently. Never claim real industrial performance from this dataset.
- Never generate or display invented metrics as measured. Toy fixture only for unit tests; NEVER submit it as actual evaluation.
- Do not attribute true causal explanations, actual failures, or maintenance costs to unverified ML output. ML predictions != ground truth.

## Architecture
1. UCI sensor signal extraction (`factorymind/data.py`) --> per-cycle features.
2. scikit-learn model + metrics (`factorymind/train.py`).
3. `factorymind/core.py`: factual case data and provider-agnostic OpenAI-compatible chat client.
4. `factorymind/api.py`: FastAPI endpoints `/health`, `/cycles`, `/predict/{cycle_id}`, `/investigate`.
5. `app.py`: Streamlit visual demo, raw sensor trace, investigate and human inspection approval.
6. `factorymind/reviews.py`: SQLite review audit. **No actuator commands.**
7. LLM local dev: Ollama `llama3.2` via `http://localhost:11434/v1`; final AMD: cloud GPU vLLM ROCm via securely configured OpenAI-compatible URL.

## Immediate priorities (timebox: 24 hours)
P0 — Get dataset download and end-to-end baseline working with **real UCI data**; test feature/label alignment. Run `pytest -q` and `python -m factorymind.train`; report real metrics, dataset counts, and reproducible commands.
P0 — Run Streamlit UI with real predictions, raw charts, fallback if Ollama unavailable; ensure one full working case flow.
P0 — Investigate end-to-end with Ollama; build bounded, evidence-grounded prompts, no invented sensor attribution, and clearly marked unsupported claims.
P1 — Add automated API tests, UX polish (good and faulty case selection), robust error messages, measured latency, and review audit.
P1 — Coordinate cloud AMD vLLM inference; until cloud credits are approved, prepare exact verified operator steps but do not claim AMD has run.
P2 — Improve validation (blocked splits / group validation), compare models and calibrate confidence or rename it as uncalibrated score; add explainability (permutation importance or SHAP if feasible).
P2 — README, demo scenario, pitch outline, transparent business impact methodology and dataset/license attributions.

## How to work without hand-holding
- Inspect code before editing. Preserve working behavior. Take tasks end to end, including fixes.
- When uncertain, choose the simplest reliable implementation and document tradeoffs rather than repeatedly asking the owner.
- Don't install five agent frameworks, add RAG without a real need, or overbuild infrastructure.
- Keep modifications small, test them, report factual results and blockers.
- User-facing status should be concise: what was built, what passed, what is blocked, what human approval is needed.
- Never commit `.env`, credentials, secrets, raw archives, models, data copies, or other sensitive content.
- Track any MIT license for **our code** separately from the dataset's CC BY 4.0 license. Check competition terms and third-party licenses before publishing.

## Definition of Done — local V1
- `pytest -q` passes; dataset and model code execute on real UCI files.
- `streamlit run app.py` allows selection of a cycle, reveals sensor signals, a model-generated prediction, evidence-backed commentary (or transparent fallback), and records approval/rejection.
- `uvicorn factorymind.api:app --port 8001` returns a prediction and coherent investigation response.
- At least one correct and one incorrect predicted case described honestly in demo or evaluation, plus complete confusion matrix and per-class metrics.
- No fabricated measurements, cost savings, machine IDs, time stamps, or AMD runtime evidence.

## Definition of Done — contest submission (separate later milestone)
- AMD GPU verified (`rocm-smi` or available tooling), actual vLLM inference served from AMD infrastructure and invoked by the demonstrated product.
- Show provenance evidence: AMD instance type (verified), model, server configuration, non-secret health checks, measured latency, and actual demo request/response.
- Secure endpoint; do not expose unprotected vLLM APIs publicly.
- Public GitHub repository, README that runs, working deployment URL, recorded demo, pitch slides, cover image, dataset attribution, AI-assistance disclosure if required.
- Human owner explicitly reviews and performs the actual submission.
