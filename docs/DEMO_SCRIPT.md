# FactoryMind AI — local demonstration

Run Streamlit and FastAPI using the README. Keep Ollama available for the live investigation. All demonstrations below use the retained random-split model and held-out cases.

1. **Frame the problem:** “This prototype helps an operator move from a valve-condition prediction to an evidence-backed inspection review. These are public experimental rig recordings, not a live plant.”
2. **Normal case:** choose “Held-out · optimal predictions”, cycle **4**. Reveal the annotation: predicted optimal, dataset label optimal. Explain that the score is uncalibrated.
3. **Fault-class case:** choose “Held-out · non-optimal predictions”, cycle **213**. Predicted near failure, dataset label near failure. Show pressure and flow traces alongside optimal-label cycle 4. The reference differs in other rig conditions and is not an operating threshold. Do not claim a causal sensor signature.
4. **Live AI:** click “Investigate selected cycle”. The result must say AI-assisted and name the configured model. Explain: “The LLM selects review focus and checks; the app validates its choices and inserts the actual measurements.” If it says fallback, say plainly that no AI output was used.
5. **Human review:** enter “DEMO — no real inspection requested”, then click **Approve inspection review**. Show the stored review and real review timestamp. Click **Reject inspection review** to demonstrate the alternate path if useful. Neither controls equipment or creates an external work order.
6. **Show an error:** choose “Held-out · model errors”, cycle **248**. The model predicts near failure, but the dataset label is optimal. “This error demonstrates why a model output cannot authorize maintenance or be treated as ground truth.”
7. **Show evaluation:** random holdout accuracy 98.01%; blocked accuracy 82.86%. Show both confusion matrices and note blocked small-lag recall of 48.89%. Explain neighboring-cycle correlation and lack of independent-plant validation.
8. **AMD statement:** “Local Ollama integration is verified. The same investigation interface is prepared for a secure AMD ROCm vLLM server, but AMD execution is pending approved access.” Replace this only after collecting genuine server and product evidence.

Suggested pitch outline: operator problem → recorded sensor evidence → human decision workflow → honest evaluation including an error → demonstrated inference provenance → next validation milestone. Do not present a contest score or claim prize eligibility.

Business impact remains a hypothesis. A future study should compare the same qualified reviewers with and without the tool using matched cases, recording review time, decision quality, false positives and missed conditions. Randomize order to reduce learning effects. Only calculate potential labor savings from measured time differences and an explicitly sourced cost assumption. Do not infer downtime savings from classifier accuracy.
