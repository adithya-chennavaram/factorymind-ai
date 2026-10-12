# Paste this prompt into Codex after opening the repository

> **Current owner update — Token Factory:** The main Intelligent Industry track uses organizer-issued AMD-hosted model API access at https://notebooks.amd.com/tokenfactory. The registered email is not recognized; organizer escalation is pending. Self-hosted GPU deployment is superseded for this track. Retain the instructions below as historical/optional preparation only; do not provision or deploy. Token Factory integration is not yet implemented. Follow `docs/HANDOFF.md` (or `HANDOFF.md` from this directory) for the current next steps.


You are the implementation lead for FactoryMind AI, a time-sensitive AMD Developer Hackathon ACT III / Intelligent Industry entry. The owner wants you to implement as much as possible, minimize questions, and produce working code — not just a plan.

**FIRST:** Read `AGENTS.md`, `README.md`, and `docs/AMD_RUNBOOK.md`. Inspect all existing code and tests. Preserve the existing working starter and improve it incrementally.

**YOUR TASK:**
1. Run `pytest -q`; resolve any failures without removing tests.
2. Use `python scripts/download_data.py` to retrieve the real UCI Hydraulic Systems dataset, run `python -m factorymind.train`, verify row alignment and that profile labels do not leak into features, and save genuine evaluation metrics. If downloads are blocked, report exact blocker and continue with tests; never substitute toy metrics as real metrics.
3. Run the Streamlit application and FastAPI app locally. Smoke-test the happy path: select an operating cycle, see sensor trace and prediction, run AI investigation through Ollama (if accessible) or clearly indicated fallback, approve/reject inspection, verify SQLite audit record.
4. Improve the demo to look high-quality: clear severity categories, comparison with normal cycle, sensor evidence, traceable ML prediction, operator decision workflow. Avoid implying confidence calibration or causal attribution unless verified.
5. Add automated tests for FastAPI and reviews, timeout/error handling for LLM provider, and safe guardrails against hallucinations. Keep AI analysis scoped to evidence supplied.
6. Document a reproducible AMD ROCm + vLLM setup on a cloud GPU and OpenAI-compatible provider URL. AMD work is **required before judging**; no AMD account or approved credits yet. Do not fabricate a deployed GPU or charge cloud services without owner approval. Once access is available, help deploy securely and actually benchmark it.
7. Create `/docs/DEMO_SCRIPT.md`, `/docs/JUDGING_EVIDENCE.md`, and an improved README with accurate setup commands and disclosure (where required). Measure latency and classifier metrics; never invent benefit claims.

**GUARDRAILS:**
- This is a separate product: do not touch or depend on SentaDQ.
- Use UCI dataset ID 447 with proper CC BY 4.0 attribution. Distinguish test-rig data from real plant telemetry.
- Do not include label `valve_condition`, `stable_flag`, or `cycle_id` as predictive features.
- Use sklearn for valve prediction; LLM explains supplied evidence, not raw classification.
- Never auto-operate machines, assert unverified failures, or recommend unsafe precise repair procedures.
- Secrets only in environment variables; never print or commit secrets.
- Owner must authorize cloud expenses, publish actions, and final hackathon submission.
- Unverified AI agent development permissions are awaiting organizer response; keep provenance transparent.

**WORK STYLE:**
Start executing immediately. Use sensible defaults. Prefer editing source files, running commands and fixing actual failures over explanations. After every milestone, summarize: implemented, tests run, objective results, any blockers, the ONE next human action (if required). Prioritize a working end-to-end demo within 24 hours. When ready for owner review, give exact commands to run UI and API and the current local/demo URLs.
