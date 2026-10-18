# FactoryMind AI — 2–3 Minute Judge Demo

Open the [hosted Streamlit app](https://factorymind-ai-eht5zkagrajhz6m7dza5kv.streamlit.app/) and choose **Model Trust Lab** from the sidebar. The hosted instance may take a few minutes on a cold start while it downloads the public UCI files and creates ignored evaluation artifacts.

1. Start with the **accuracy illusion** cards. Explain that the random stratified holdout is **98.01%**, while the retrospective five-fold blocked evaluation is **82.86%**. The difference is the reason to audit trust before relying on a green result.
2. Set the review budget to **20%**. The blocked audit contains **1,344 predicted-green cycles** and **260 false-green predictions**. Confidence-only review inspects **266** cases and identifies **136** of those false-green cases. Stress-only and the original combined score performed worse in the retained experiment.
3. Choose **cycle 2053** for a stable-looking but wrong green prediction, or **cycle 507** for a case whose class changes in simulated sensor-gain scenarios. Explain that these are retrospective UCI test-rig examples, not factory incidents.
4. Ask the **AMD Evidence Hearing** to prioritize bounded evidence. The model receives the case packet without the held-out label and can select only approved evidence IDs, action IDs and verdicts. Numerical values are rendered locally from validated artifacts.
5. Review the selected evidence and human checks. The provider label identifies AMD Token Factory when the request succeeds. If the shared quota is unavailable, continue with the local evidence; fallback output is explicitly labeled rule-based and is never called AMD-generated.
6. Approve or reject the inspection review. Enter an optional operator note. The decision is stored in local SQLite with cycle, blocked fold, review policy, stress findings, provider/model, validated selection, UTC timestamp and evidence checksum.
7. Open the saved review and download the **Decision Passport**. Confirm the checksum status. Then reveal the held-out label for evaluation. The passport intentionally excludes that label before the decision.

Use this closing sentence: **“FactoryMind does not claim that a model found a physical failure. It shows when a green prediction deserves human review, records what the evidence was, and keeps the operator in control.”**

The dataset is an experimental hydraulic test rig, not live factory telemetry. Stress tests are hypothetical gain perturbations, scores are uncalibrated, blocked validation is retrospective, and no future failure prevention, factory savings or safety guarantee has been established.
