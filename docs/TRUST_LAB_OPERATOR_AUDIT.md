# Model Trust Lab — operator approval and decision evidence

This patch connects the existing **blocked out-of-fold Model Trust Lab** and live AMD Token Factory Evidence Hearing to a local, human-only decision workflow.

## What changes

- An **Approve inspection review** / **Reject inspection review** form appears **after a successful AMD Evidence Hearing**, on the same Trust Lab cycle and review budget.
- Each review is stored in `trust_lab_reviews` inside the existing SQLite database (`FACTORYMIND_REVIEW_DB` or `data/reviews.sqlite3`). The existing main-dashboard `reviews` table remains unchanged. Separate tables matter: cycle 507 is predicted **Optimal** by the fold-specific model, but **Small lag** by the main app's distinct random-split model.
- The stored decision includes cycle, held-out fold, **unrounded blocked-model healthy score**, policy and budget, simulated gain measurements, SHA-256 of the scored CSV, the *validated bounded* AMD model reply and displayed recommendations, time and optional operator note.
- Ground truth is **never passed to AMD and never included in the pre-reveal decision evidence**. Operators can still reveal the dataset annotation separately for retrospective education.
- Review history is visible on the same page; select a decision to inspect/download its JSON evidence passport. The stored evidence checksum detects accidental edits to the JSON but **is not proof of immutability**, nor certification of the underlying dataset or model.
- The main dashboard footer now reflects the available AMD API integration rather than saying it is pending.

## Apply safely

Unzip this patch over the project root. It adds a new `factorymind/trust_audit.py` module, a new test file, and updates the Model Trust Lab page and helper. No destructive database migration is performed and **no dataset, secrets, model weights or existing review data are included**.

## Test locally

```bash
cd ~/Downloads/factorymind-ai
python -m pytest -q
python -m streamlit run app.py --server.port 8502
```

1. In **Model Trust Lab**, choose **Cycle 507**, review budget **20%**.
2. Click **Ask AMD to prioritize evidence for this case**. The AMD team API key must already be set in the terminal that launches Streamlit.
3. After a validated response, enter an optional operator note and click **Approve inspection review**.
4. Confirm the saved decision and its **blocked-fold provenance** appear under **Trust Lab audit history for this cycle**.
5. Open **Inspect saved decision evidence** and download the JSON passport if desired.
6. Finally click **Reveal held-out dataset label**. The correct retrospective label is **Small lag**. This must not be confused with the separately trained main app prediction for the same cycle.

## Important limitations

- A successful review records a **request for human inspection only**; no actuator or work-order integration exists.
- Stress-test simulated gains are not certified sensor-noise distributions, and model stability does not mean correctness.
- The current confidence-only review policy is supported by retrospective blocked out-of-fold measurements; results do **not** prove future deployment benefit.
- AMD Token Factory inference is only used when a user clicks the hearing button and the provider is configured. Do not publish API credentials.
