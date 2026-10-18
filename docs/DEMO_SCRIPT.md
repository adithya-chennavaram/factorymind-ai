# FactoryMind AI — Model Trust Lab demonstration

Reproduce the Trust Lab artifacts and configure Token Factory using the README. Use the Model Trust Lab page for this sequence:

1. Show random holdout accuracy of 98.01% alongside blocked validation accuracy of 82.86%. Explain that neither establishes factory performance.
2. Show 260 false-green predictions among 1,344 predicted-green cycles. At the 20% per-fold review budget, confidence-only ranking catches 136 using 266 reviews.
3. Inspect cycle 2053: predicted Optimal, 88.6% uncalibrated score, 0/50 class switches under hypothetical sensor-gain scenarios. Reveal Small lag: stable can still be wrong.
4. Inspect cycle 507: predicted Optimal, 41.6% score, 26/50 hypothetical class switches. These are simulated gain changes, not observed equipment failures.
5. Run the AMD evidence hearing only with authorized quota. Show the actual returned model and accepted bounded recommendations. A failed request must not be described as AMD-generated output.
6. Explicitly approve an inspection review, show the recorded decision, and export the checksum-bearing Decision Passport. No equipment is controlled.
7. Reveal the held-out Small lag label after the decision; it was excluded from the hearing evidence. Close with the limits of retrospective experimental validation.

The published recording demonstrates this sequence. It is available in the [submission release](https://github.com/adithya-chennavaram/factorymind-ai/releases/tag/act-iii-submission). Prospective safety, downtime prevention and cost savings remain unverified.
