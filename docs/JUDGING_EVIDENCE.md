# Submission evidence

- **Dataset:** 2,205 UCI hydraulic test-rig cycles, 30 sensor features. Experimental rig data, not live factory telemetry.
- **Classifier:** random holdout accuracy 98.01%; five blocked folds 82.86%. See [full evaluation](EVALUATION.md) and [misclassification analysis](BLOCKED_ERROR_ANALYSIS.md).
- **Trust Lab:** 1,344 blocked out-of-fold predictions marked Optimal include 260 incorrect green predictions. Confidence-only review catches 136 at a 266-case review budget. Simulated gain tests are hypothetical, not physical robustness validation.
- **AMD:** the released recording shows Token Factory returning `self-dploy/DeepSeek-V4-Flash`, followed by inspection approval #3 for cycle 507 and a checksum-verified Decision Passport. Managed-service provenance is documented; physical GPU identity is not independently attested.
- **Human control:** no actuator commands, automatic approval or external work orders. Ground-truth reveal follows the recorded decision.
- **Release:** [public source](https://github.com/adithya-chennavaram/factorymind-ai) and [video, pitch, cover and subtitles](https://github.com/adithya-chennavaram/factorymind-ai/releases/tag/act-iii-submission).

There is no hosted interactive deployment. Actual hackathon submission and organizer confirmation of eligibility/AI-assistance rules remain the owner's responsibility. See README for the current test result, reproduction commands, dataset license and development disclosure.
