# Local V1 delivery checklist

Verified October 12, 2026. See JUDGING_EVIDENCE.md for actual measurements and limits.

## Within first 2 hours
- [x] Inspect existing files, install requirements and pass unit tests
- [x] Download official UCI raw files, build feature CSV
- [x] Run actual model training and capture real evaluation report
- [x] Identify class imbalance and potential validation leakage

## Within 6 hours
- [x] Streamlit displays a real cycle and sensor trace
- [x] Trained classification can distinguish 100/90/80/73 labels
- [x] API `/predict/{cycle_id}` works and is tested

## Within 12 hours
- [x] Ollama (if accessible) creates a cautious, grounded investigation
- [x] Review approval/rejection is saved locally
- [x] Full demo flow can run twice without errors
- [x] FastAPI unit/integration tests expanded

## By 24 hours
- [x] Improved design and evidence-to-decision flow
- [x] Model metrics and error cases documented
- [x] Demo script drafted and short screen recording possible
- [x] AMD GPU approval/access blocker visible and one human action stated

## Submission blockers that cannot be faked
- AMD cloud GPU availability and approval
- AMD vLLM deployment and actual successful endpoint test
- User approval for any paid deployment or public push
- AI assistant permission and disclosure rules confirmed by organizers
- Actual deployed public URL, video and slides
