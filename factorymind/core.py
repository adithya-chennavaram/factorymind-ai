"""Shared inference and bounded LLM-assisted investigation."""
import json
from pathlib import Path
from time import perf_counter
from typing import Literal
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from .data import TARGET_LABELS, SENSOR_INFO, feature_columns
from .providers import ProviderConfig, ModelMismatchError, hardware_status, error_category

ROOT = Path(__file__).resolve().parent.parent
CHECKS = {
    "review_trace": "Have a qualified operator review the recorded sensor trace and acquisition quality.",
    "compare_reference": "Compare with a dataset cycle labeled optimal; this is a reference, not a validated operating threshold.",
    "review_procedures": "Consult documented inspection procedures before deciding whether an inspection is warranted.",
    "verify_prediction": "Independently verify the predicted valve class; the model can be wrong.",
}


def load_assets(root: Path = ROOT):
    model_path = root / "artifacts/valve_model.joblib"
    features_path = root / "data/processed/features.csv"
    if not model_path.is_file() or not features_path.is_file():
        raise FileNotFoundError("Model/features missing. Run python -m factorymind.train first.")
    model = joblib.load(model_path)
    metrics_path = root / "artifacts/metrics.json"
    model.factorymind_metadata_ = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return model, pd.read_csv(features_path)


def predict_cycle(model, features: pd.DataFrame, cycle_id: int) -> dict:
    start = perf_counter()
    selection = features.loc[features["cycle_id"] == cycle_id]
    if selection.empty:
        raise ValueError(f"Unknown cycle_id: {cycle_id}")
    row = selection.iloc[[0]]
    probs = model.predict_proba(row[feature_columns(features)])[0]
    clf = model.named_steps["classifier"]
    label = int(clf.classes_[probs.argmax()])
    evidence = {s: {"mean": round(float(row[f"{s}_mean"].iloc[0]), 3),
                    "std": round(float(row[f"{s}_std"].iloc[0]), 3), "unit": SENSOR_INFO[s][1]}
                for s in SENSOR_INFO}
    metadata = getattr(model, "factorymind_metadata_", {})
    return {"cycle_id": int(cycle_id), "predicted_valve_condition_code": label,
        "predicted_valve_condition": TARGET_LABELS[label],
        "model_confidence_uncalibrated": round(float(probs.max()), 4),
        "sensor_evidence": evidence, "source": "UCI hydraulic experimental test rig data",
        "evaluation_split": str(row["evaluation_split"].iloc[0]) if "evaluation_split" in row else "unknown",
        "dataset_label_code": int(row["valve_condition"].iloc[0]),
        "dataset_label": TARGET_LABELS[int(row["valve_condition"].iloc[0])],
        "model_version": metadata.get("model_sha256", "unversioned"),
        "prediction_latency_ms": round((perf_counter() - start) * 1000, 2),
        "limits": "Prediction is not a verified diagnosis. Sensor summaries are observations, not causal attributions. Score is uncalibrated."}


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_sensors: list[Literal["PS1", "PS2", "FS1", "TS1", "VS1"]] = Field(min_length=1, max_length=5)
    interpretation: Literal["optimal_prediction", "non_optimal_prediction", "insufficient_evidence"]
    human_checks: list[Literal["review_trace", "compare_reference", "review_procedures", "verify_prediction"]] = Field(min_length=1, max_length=4)


def investigation_prompt(case: dict) -> list[dict[str, str]]:
    system = (
        "You are a cautious industrial maintenance decision-support assistant. Use ONLY supplied evidence. "
        "Do not fabricate causes, thresholds, measurements, instructions, or safety assurances. "
        "Your task is to select sensor observations for human review, interpret the model prediction, "
        "and choose safe inspection checks. Sensor choice is review focus, NOT feature importance or causality. "
        "Return ONLY a JSON object with keys evidence_sensors (one to five of PS1, PS2, FS1, TS1, VS1), "
        "interpretation (optimal_prediction if predicted code is 100, non_optimal_prediction otherwise, "
        "or insufficient_evidence), human_checks (one to four of review_trace, compare_reference, "
        "review_procedures, verify_prediction). No extra keys or prose. Model score is uncalibrated. "
        "Do not authorize equipment operation."
    )
    evidence_case = {k: case[k] for k in ("cycle_id", "predicted_valve_condition_code", "predicted_valve_condition",
        "model_confidence_uncalibrated", "sensor_evidence", "source", "limits")}
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(evidence_case)}]


def render_plan(plan: InvestigationPlan, case: dict) -> str:
    expected = "optimal_prediction" if case["predicted_valve_condition_code"] == 100 else "non_optimal_prediction"
    if plan.interpretation not in (expected, "insufficient_evidence"):
        raise ValueError("LLM interpretation contradicts supplied prediction")
    facts = []
    for sensor in dict.fromkeys(plan.evidence_sensors):
        e = case["sensor_evidence"][sensor]
        facts.append(f"- {sensor}: mean {e['mean']} {e['unit']}; standard deviation {e['std']} {e['unit']}.")
    interpretation = ("Evidence is insufficient for a diagnosis." if plan.interpretation == "insufficient_evidence"
                      else f"The classifier predicts **{case['predicted_valve_condition']}**. This is a hypothesis for human verification.")
    return ("### EVIDENCE\n" + "\n".join(facts) + "\n\n### POSSIBLE EXPLANATIONS\n" + interpretation
        + " No physical cause is established by these observations.\n\n### SUGGESTED HUMAN CHECKS\n"
        + "\n".join("- " + CHECKS[c] for c in dict.fromkeys(plan.human_checks))
        + "\n\n### UNCERTAINTY\nThe model score is uncalibrated, not a failure probability. "
          "AI selected the review focus and checks; factual wording comes from validated data and a bounded catalog. "
          "Selected sensors are not causal attributions. Experimental rig results do not establish plant performance.")


def investigate(case: dict, *, config=None, telemetry=None) -> str:
    config = config or ProviderConfig.from_env()
    telemetry = telemetry if telemetry is not None else {}
    options = {}
    if config.disable_thinking:
        options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    start = perf_counter()
    with config.client() as client:
        completion = client.chat.completions.create(model=config.model,
            messages=investigation_prompt(case), temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p,
            response_format={"type": "json_object"}, **options)
    telemetry["provider_latency_ms"] = round((perf_counter() - start) * 1000, 3)
    usage = getattr(completion, "usage", None)
    for field, name in [("prompt_tokens", "prompt_tokens"), ("completion_tokens", "generated_tokens")]:
        count = getattr(usage, field, None)
        telemetry[name] = count if isinstance(count, int) and count >= 0 else None
    returned_model = getattr(completion, "model", None)
    # Preserve the actual server-reported model even when validation fails.
    # The response still falls back safely, and this metadata does not contain API keys.
    telemetry["response_model"] = returned_model
    if not config.matches_model(returned_model):
        raise ModelMismatchError("Provider returned a different model")
    telemetry["finish_reason"] = completion.choices[0].finish_reason if completion.choices else None
    if not completion.choices or completion.choices[0].finish_reason != "stop":
        raise ValueError("Incomplete provider response")
    plan = InvestigationPlan.model_validate_json(completion.choices[0].message.content or "")
    text = render_plan(plan, case)
    telemetry["validated_plan"] = plan.model_dump()
    return text


def safe_fallback(case: dict, error: Exception) -> str:
    return ("LLM unavailable or returned an invalid response. **Rule-based fallback — not an AI investigation.**\n\n"
        f"Model prediction: **{case['predicted_valve_condition']}**; uncalibrated score, not a diagnosis.\n\n"
        "A qualified operator should examine the recorded sensor signals and documented procedures, "
        "independently verify the prediction, and decide whether inspection is needed. No equipment control.\n\n"
        f"Failure category: {type(error).__name__} (provider details omitted).")


def run_investigation(case: dict, *, config=None) -> dict:
    start = perf_counter()
    telemetry = {"response_model": None, "generated_tokens": None, "prompt_tokens": None,
                 "provider_latency_ms": None, "finish_reason": None}
    failure = None
    try:
        config = config or ProviderConfig.from_env()
        text = investigate(case, config=config, telemetry=telemetry)
        mode = "llm"
    except Exception as exc:
        text, mode, failure = safe_fallback(case, exc), "fallback", error_category(exc)
    elapsed = (perf_counter() - start) * 1000
    served_by = ({"ollama": "ollama", "amd_vllm": "vllm", "amd_tokenfactory": "amd_tokenfactory"}[config.provider]) if mode == "llm" else "rule_based"
    # A fallback never inherits the attempted server's hardware attribution.
    hardware = hardware_status(config, telemetry["response_model"]) if mode == "llm" else {
        "status": "not_applicable", "hardware_verified": False, "gpu_details": None,
        "note": "Rule-based fallback; no AI output used."}
    label = {"ollama": "Local Ollama", "vllm": "vLLM · GPU unverified",
             "amd_tokenfactory": "AMD Token Factory · AMD-hosted API (documented)",
             "rule_based": "Rule-based fallback"}[served_by]
    if served_by == "vllm" and hardware["status"] == "operator_evidence_matched":
        label = "vLLM · matching operator AMD evidence"
    tokens = telemetry["generated_tokens"]
    return {"case": case, "investigation": text, "provider_mode": mode, "served_by": served_by,
            "provider_label": label, "attempted_provider": config.public() if config else {"status": "invalid_configuration"},
            "model": telemetry["response_model"] if mode == "llm" else None,
            "hardware": hardware, "latency_ms": round(elapsed, 3), "failure": failure,
            "telemetry": telemetry,
            "generated_tokens_per_second_e2e": round(tokens / (elapsed / 1000), 3) if tokens is not None and elapsed > 0 else None,
            "requires_human_approval": True, "equipment_control": False,
            "guardrail": "Validated bounded AI selection; no free-form generated measurements or actions."}
