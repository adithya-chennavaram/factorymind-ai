"""Local API serving the same real-data pipeline as Streamlit."""
from functools import lru_cache
import sqlite3
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from .core import ROOT, load_assets, predict_cycle, run_investigation
from .data import SENSOR_INFO, read_cycle_samples
from .reviews import save_review, list_reviews
from .providers import ProviderConfig, ProviderConfigError, check_connection

app = FastAPI(title="FactoryMind AI", version="0.2.0")
REVIEW_PATH = ROOT / "data/reviews.sqlite3"


@app.get("/provider/health")
def provider_health():
    try:
        result = check_connection(ProviderConfig.from_env())
    except ProviderConfigError as exc:
        raise HTTPException(503, "Invalid provider configuration") from exc
    if not result["ready"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=result)
    return result


@lru_cache(maxsize=1)
def assets():
    return load_assets()


def ready_assets():
    try:
        return assets()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(503, "Model assets unavailable; run python -m factorymind.train") from exc


@app.get("/health")
def health():
    model, features = ready_assets()
    return {"service": "factorymind-ai", "status": "ok", "cycles": len(features),
            "model_version": getattr(model, "factorymind_metadata_", {}).get("model_sha256", "unversioned")}


@app.get("/cycles")
def cycles(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    _, features = ready_assets()
    return {"cycles": features.cycle_id.iloc[offset:offset + limit].astype(int).tolist(), "total": len(features)}


@app.get("/predict/{cycle_id}")
def predict(cycle_id: int):
    try:
        return predict_cycle(*ready_assets(), cycle_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/sensors/{cycle_id}/{sensor}")
def sensor_trace(cycle_id: int, sensor: Literal["PS1", "PS2", "FS1", "TS1", "VS1"]):
    predict(cycle_id)
    try:
        values = read_cycle_samples(ROOT / "data/raw", sensor, cycle_id)
    except (OSError, ValueError, IndexError) as exc:
        raise HTTPException(503, "Raw sensor trace unavailable") from exc
    step = max(1, len(values) // 600)
    return {"cycle_id": cycle_id, "sensor": sensor, "unit": SENSOR_INFO[sensor][1],
            "sample_count": len(values), "display_stride": step,
            "seconds": [i / SENSOR_INFO[sensor][2] for i in range(0, len(values), step)],
            "values": values[::step].tolist()}


class InvestigationRequest(BaseModel):
    cycle_id: int = Field(gt=0)


@app.post("/investigate")
def investigate_api(request: InvestigationRequest):
    return run_investigation(predict(request.cycle_id))


class ReviewRequest(BaseModel):
    cycle_id: int = Field(gt=0)
    decision: Literal["approve_inspection", "reject_inspection"]
    notes: str = Field(default="", max_length=2000)


@app.post("/reviews", status_code=201)
def review(request: ReviewRequest):
    case = predict(request.cycle_id)
    try:
        review_id = save_review(request.cycle_id, request.decision, request.notes, REVIEW_PATH, evidence=case)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Review storage unavailable") from exc
    return {"review_id": review_id, "decision": request.decision, "equipment_control": False}


@app.get("/reviews/{cycle_id}")
def reviews(cycle_id: int):
    predict(cycle_id)
    try:
        return {"reviews": list_reviews(cycle_id, REVIEW_PATH)}
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Review storage unavailable") from exc
