"""FactoryMind AI: evidence-to-inspection operator demonstration."""
import json
import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from factorymind.core import ROOT, load_assets, predict_cycle, run_investigation
from factorymind.data import SENSOR_INFO, TARGET_LABELS, read_cycle_samples
from factorymind.reviews import save_review, list_reviews

st.set_page_config(page_title="FactoryMind AI · Model Trust Lab", page_icon="🏭", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1400px}
[data-testid="stMetric"] {background:#142233;border:1px solid #293c50;border-radius:12px;padding:18px}
h1 {letter-spacing:-1.5px} [data-testid="stSidebar"] {border-right:1px solid #293c50}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def assets():
    return load_assets()

@st.cache_data
def trace(sensor, cycle_id):
    return read_cycle_samples(ROOT / "data/raw", sensor, cycle_id)

try:
    model, features = assets()
    metrics = model.factorymind_metadata_
    predictions = pd.read_csv(ROOT / "artifacts/predictions.csv")
except (FileNotFoundError, OSError, ValueError) as exc:
    st.title("FactoryMind AI — Model Trust Lab")
    st.warning(f"Local assets are not ready: {type(exc).__name__}")
    st.code("python scripts/download_data.py\npython -m factorymind.train", language="bash")
    st.stop()

held = predictions[predictions.split == "held_out"]
collections = {"Held-out · non-optimal predictions": held[held.predicted != 100],
               "Held-out · optimal predictions": held[held.predicted == 100],
               "Held-out · model errors": held[held.predicted != held.actual],
               "All held-out cycles": held, "All cycles (includes training)": predictions}
with st.sidebar:
    st.markdown("## FACTORYMIND AI\nModel Trust Lab")
    st.caption("LOCAL PROTOTYPE · HUMAN REVIEW")
    collection = st.selectbox("Case collection", list(collections))
    choices = collections[collection].cycle_id.astype(int).tolist()
    if not choices:
        st.info("No cases in this collection.")
        st.stop()
    cycle_id = st.selectbox("Operating cycle", choices)
    st.divider()
    st.markdown("**Data provenance**")
    st.write(f"{len(features):,} experimental cycles · 60 seconds each · 5 sensors · 30 features")
    st.caption("UCI Hydraulic Systems, CC BY 4.0. Recorded test-rig cycles; not live factory telemetry.")
    st.markdown("[Dataset and attribution](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)")
    st.caption("Local review audit only. No actuator connection.")

case = predict_cycle(model, features, int(cycle_id))
st.caption(f"VALVE CONDITION / CYCLE {cycle_id:04d} / {case['evaluation_split'].replace('_', ' ').upper()}")
st.title("From sensor evidence to human decision")
st.write("Inspect a model prediction, investigate recorded signals, and decide whether to request an inspection.")
a, b, c = st.columns([2, 1, 1])
a.metric("Predicted valve condition", case["predicted_valve_condition"])
b.metric("Uncalibrated model score", f"{case['model_confidence_uncalibrated']:.1%}")
c.metric("Prediction latency", f"{case['prediction_latency_ms']:.0f} ms")
if case["predicted_valve_condition_code"] != 100:
    st.warning("Non-optimal switching predicted. The class name describes the model output; it does not establish a physical failure.")
else:
    st.info("Optimal switching predicted. This prediction is not a safety assurance.")

sensor_tab, evaluation_tab = st.tabs(["01 · Sensor evidence & investigation", "02 · Evaluation & limitations"])
with sensor_tab:
    st.subheader("Recorded sensor evidence")
    left, right = st.columns([2, 1])
    with left:
        sensor = st.selectbox("Sensor trace", list(SENSOR_INFO), format_func=lambda s: f"{s} · {SENSOR_INFO[s][0]} ({SENSOR_INFO[s][1]})")
        reference_id = int(held[held.actual == 100].cycle_id.iloc[0])
        compare = st.checkbox(f"Compare with cycle {reference_id} (dataset label: optimal)", value=True)
        try:
            sample = trace(sensor, int(cycle_id))
            stride = max(1, len(sample) // 600)
            data = {f"Selected cycle {cycle_id}": sample[::stride]}
            if compare:
                data[f"Optimal-label reference {reference_id}"] = trace(sensor, reference_id)[::stride]
            chart = pd.DataFrame(data, index=np.arange(0, len(sample), stride) / SENSOR_INFO[sensor][2])
            chart.index.name = "Seconds into cycle"
            st.line_chart(chart, x_label="Seconds into 60-second cycle", y_label=SENSOR_INFO[sensor][1], height=300)
            st.caption(f"{len(sample):,} real samples at {SENSOR_INFO[sensor][2]} Hz; display uses every {stride} sample(s). Reference is not a validated threshold; other rig conditions may differ.")
        except (OSError, IndexError, ValueError):
            st.warning("Raw trace unavailable. Sensor summary values remain available.")
    with right:
        st.dataframe(pd.DataFrame(case["sensor_evidence"]).T.rename(columns={"std": "std deviation"}), width="stretch")
        st.caption("Means and standard deviations over the full cycle. These are observations, not model attribution or causal evidence.")
    with st.expander("Reveal dataset annotation and prediction provenance"):
        st.write(f"Dataset label: **{case['dataset_label']}** · Prediction: **{case['predicted_valve_condition']}**")
        st.write("Prediction matches dataset label." if case["dataset_label_code"] == case["predicted_valve_condition_code"] else "Prediction is incorrect against the dataset label. Include this failure in the demo.")
        st.caption(f"Split: {case['evaluation_split']}. Labels, stability and cycle ID are excluded from the 30 predictors. Dataset label is withheld from the LLM.")
        st.code("Model SHA-256: " + case["model_version"])
    st.subheader("AI investigation")
    st.caption("The LLM selects a review focus and safe checks. Responses are validated against a bounded schema; measurements come directly from the case.")
    if st.button("Investigate selected cycle", type="primary"):
        with st.spinner("Requesting and validating investigation…"):
            st.session_state["investigation"] = run_investigation(case)
    stored = st.session_state.get("investigation")
    if stored and "served_by" not in stored:
        st.session_state.pop("investigation", None)
        stored = None
        st.info("Provider attribution was upgraded. Run a new investigation to capture its source.")
    if stored and stored["case"]["cycle_id"] == int(cycle_id):
        if stored["provider_mode"] == "llm":
            st.success(f"{stored['provider_label']} · {stored['model']} · {stored['latency_ms'] / 1000:.2f} s")
            st.caption(stored['hardware']['note'])
        else:
            st.warning(f"Rule-based fallback · {stored['latency_ms'] / 1000:.2f} s · No AI output used")
        st.caption(f"Request endpoint ID: {stored['attempted_provider'].get('endpoint_id', 'unavailable')} · Failure: {stored['failure'] or 'none'}")
        st.markdown(stored["investigation"])
        st.download_button("Download investigation evidence", json.dumps(stored, indent=2), file_name=f"cycle_{cycle_id}_investigation.json", mime="application/json")
        st.subheader("Human decision · inspection request only")
        with st.form(f"review_{cycle_id}"):
            note = st.text_input("Operator note (optional)", max_chars=2000)
            approve_col, reject_col = st.columns(2)
            approved = approve_col.form_submit_button("Approve inspection review")
            rejected = reject_col.form_submit_button("Reject inspection review")
        if approved or rejected:
            try:
                review_id = save_review(int(cycle_id), "approve_inspection" if approved else "reject_inspection",
                    note, os.getenv("FACTORYMIND_REVIEW_DB", str(ROOT / "data/reviews.sqlite3")), evidence=stored)
                st.success(f"Decision #{review_id} recorded in the local audit. No machinery was controlled.")
            except (OSError, sqlite3.Error):
                st.error("The review could not be saved. Check local database permissions and retry.")
    with st.expander("Local review history for this cycle"):
        reviews = list_reviews(int(cycle_id), os.getenv("FACTORYMIND_REVIEW_DB", str(ROOT / "data/reviews.sqlite3")))
        if reviews:
            st.dataframe(pd.DataFrame(reviews), width="stretch", hide_index=True)
        else:
            st.caption("No reviews recorded for this cycle.")

with evaluation_tab:
    st.subheader("Measured performance on real UCI data")
    blocked = metrics.get("blocked", {})
    st.dataframe(pd.DataFrame([
        {"Evaluation": "Random stratified holdout", "Cycles evaluated": metrics["test_samples"], "Accuracy": metrics["accuracy"], "Macro F1": metrics["macro_f1"]},
        {"Evaluation": "5 contiguous folds · 10-cycle purge", "Cycles evaluated": blocked.get("test_samples"), "Accuracy": blocked.get("accuracy"), "Macro F1": blocked.get("macro_f1")},
        {"Evaluation": "Majority-class holdout baseline", "Cycles evaluated": metrics["test_samples"], "Accuracy": metrics["majority_baseline"]["accuracy"], "Macro F1": metrics["majority_baseline"]["macro_f1"]}
    ]), hide_index=True, width="stretch")
    st.warning("Random splitting is optimistic for neighboring rig cycles. Blocked performance is lower. Neither evaluation demonstrates generalization to another rig or a real plant.")
    selected = st.radio("Show detailed metrics", ["Random holdout", "Blocked pooled"], horizontal=True)
    report = metrics if selected == "Random holdout" else blocked
    labels = [TARGET_LABELS[c] for c in report["class_order"]]
    st.write("Confusion matrix · rows = dataset labels; columns = predictions")
    st.dataframe(pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels), width="stretch")
    st.write("Per-class precision, recall, F1 and support")
    st.dataframe(pd.DataFrame({label: report["per_class"][label] for label in labels}).T, width="stretch")
    st.caption("The served model is the original random-split training model, not a refit on all cycles. Scores are uncalibrated. No costs, downtime savings or industrial efficacy are claimed.")
st.divider()
st.caption("FactoryMind AI — Model Trust Lab · Human-approved inspection support · AMD Token Factory inference available when configured (local connection required) · Built with disclosed AI coding assistance")
