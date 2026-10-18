# FactoryMind AI — Final Public Repository Consistency Audit

Audit date: 2026-10-18
Audited branch: `main`
Audit commit before documentation corrections: `a9d93c16a68767ce9ca26b10a4bfad328f89f0de`

## Results

The audit found **13 public consistency issues** and corrected all 13 in the repository or GitHub release description. This follow-up also moved detailed setup commands into `docs/LOCAL_DEVELOPMENT.md` to keep the README judge-first. No classifier, evaluation artifact, provider implementation, deployment secret or submitted media was changed. The current test suite remains **106 passed with one upstream Starlette/httpx deprecation warning**.

| File or external resource | Old incorrect statement | New corrected statement | Reason and verification source | Status |
|---|---|---|---|---|
| `README.md` | Linked to the `fcksociety` team's submission | Links to `/factorymind-ai/factorymind-ai-model-trust-lab` | Correct URL verified with HTTP 200 | Fixed |
| `README.md` | Hosted app wording included an authentication caveat and did not clearly identify public access | Identifies the public Streamlit app and owner's incognito verification | Owner confirmation; URL retained exactly | Fixed |
| `README.md` | Did not state repository visibility clearly | States that the source repository is public and runtime/private artifacts are excluded | GitHub API reports `private: false`, `visibility: public` | Fixed |
| `docs/JUDGING_EVIDENCE.md` | “There is no hosted interactive deployment” | Links to the deployed app and submitted hackathon page | Owner incognito verification and corrected submission URL | Fixed |
| `docs/GREEN_LIGHT_BASELINE.md` | “next measurement,” approximately 269 reviews and unavailable confidence scores | Records the completed 266-review/136-catch confidence baseline and the stress/combined comparison | Current ignored evaluation artifacts and local rerun results | Fixed |
| `docs/GREEN_LIGHT_BASELINE.md` | `python scripts/evaluate_green_review.py` | `python -m scripts.evaluate_green_review` | Module execution is reproducible from the project root | Fixed |
| `docs/GREEN_CHALLENGE_EXPERIMENT.md` | Script-by-path challenge command | `python -m scripts.evaluate_green_challenge` | Module execution avoids import-path ambiguity | Fixed |
| `docs/TRUST_LAB_OPERATOR_AUDIT.md` | “Unzip this patch over the project root” | Current committed feature documentation and local run instructions | Feature is already committed and deployed | Fixed |
| `docs/COMPLETE_IMPLEMENTATION_HANDOFF.md` | Current introduction reported 102 tests | Reports 106 current tests | Latest local `pytest -q` result | Fixed |
| `docs/COMPLETE_IMPLEMENTATION_HANDOFF.md` | Hosted deployment was described as an unresolved authentication check | Records the deployed app and owner incognito verification | Owner confirmation | Fixed |
| `docs/COMPLETE_IMPLEMENTATION_HANDOFF.md` | “private-source-safe” status was ambiguous | States public source repository and excluded runtime artifacts | GitHub API visibility metadata | Fixed |
| `docs/COMPLETE_IMPLEMENTATION_HANDOFF.md` | Said README hosted-deployment wording still needed reconciliation | Removes the completed-work reminder | README is now corrected | Fixed |
| GitHub release `act-iii-submission` | Release notes said 102 current tests and no hosted app | Notes the current 106-test main branch, hosted app, real submission page and historical release snapshot | `gh release view` before/after; assets preserved | Fixed |

## Verified public resources

- GitHub API: repository is public, default branch `main`, URL `https://github.com/adithya-chennavaram/factorymind-ai`.
- Main branch after this audit is the authoritative source. Runtime credentials, raw data, model files, databases and local media remain excluded.
- Submitted page: `https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii/factorymind-ai/factorymind-ai-model-trust-lab` (HTTP 200 read-only check).
- Hosted app: `https://factorymind-ai-eht5zkagrajhz6m7dza5kv.streamlit.app/` (owner verified anonymous incognito access; unauthenticated command-line requests may receive Streamlit's auth redirect).
- Release page: `https://github.com/adithya-chennavaram/factorymind-ai/releases/tag/act-iii-submission` (HTTP 200).
- Release assets verified HTTP 200: MP4 demo, pitch PDF, editable PPTX and cover PNG. Existing assets were not replaced.
- UCI attribution and CC BY 4.0 license remain in `README.md` and `docs/EVALUATION.md`.
- MIT code license remains in `LICENSE`.
- AI coding-assistance disclosure remains in `README.md`.

## Scientific and AMD terminology check

The public documentation consistently describes 2,205 experimental UCI cycles, 30 features, 98.01% random holdout accuracy, 82.86% blocked accuracy, 1,344 predicted-green cases, 260 false-green cases and the 20% confidence-only result of 266 reviewed and 136 caught. Stress-only catches 5 and the original combined policy catches 66 at that budget; neither is presented as an improvement. The stress scenarios are hypothetical, and the blocked evaluation is retrospective.

AMD language is limited to AMD Token Factory, `DeepSeek-V4-Flash`, the observed returned identifier `self-dploy/DeepSeek-V4-Flash` and AMD-managed hosted inference. No documentation claims self-hosted MI300X/vLLM, an independently verified GPU, a ROCm benchmark or a rule-based fallback as AMD output.

## Remaining external action

The submitted lab page still exposes the post-submission metadata observed in its public HTML: a **DeepSeek V3** technology tag and the category tag **App Builder track — The INTERNET OF AGENTS**, while the selected event track shown is **Intelligent Industry** and the application itself documents DeepSeek-V4-Flash. These fields belong to the already-submitted lablab entry and were intentionally not changed during this repository-only audit. If the organizers provide an edit path, the owner should request correction of the technology tag and category; no GitHub or application change is required.

## Validation commands

```bash
git status --short
git diff --check
.venv/bin/python -m pytest -q
```

The audit did not issue a live AMD request, retrain the classifier, change the submitted video, alter repository visibility, or modify hackathon submission fields.
