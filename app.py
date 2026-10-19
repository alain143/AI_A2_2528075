"""FastAPI service: GET /health, GET /model-info, POST /predict.

Run:  uvicorn app:app --host 127.0.0.1 --port 8000
(use --host 0.0.0.0 to reach it from a phone / Expo app on the same Wi-Fi)
"""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.inference import ModelNotFoundError, Predictor
from src.validation import RecordValidationError

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    version="1.0.0",
    description="Classroom decision-support prototype (synthetic data). Human review required.",
)
# Prototype CORS so a mobile / Expo web client can call the API during development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Lazy-load the saved models; return 503 if run_all.py has not been executed."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predictor


class FarmRecord(BaseModel):
    """Typed, range-checked request body. Unknown fields are rejected (extra='forbid')."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
                "fertilizer_kg": 120.0, "field_size_ha": 1.8, "season": "A", "cooperative_support": 1,
            }
        },
    )
    rainfall_mm: Annotated[float, Field(strict=True, ge=600, le=1400)]
    avg_temp_c: Annotated[float, Field(strict=True, ge=12, le=26)]
    soil_moisture_pct: Annotated[float, Field(strict=True, ge=20, le=80)]
    fertilizer_kg: Annotated[float, Field(strict=True, ge=0, le=250)]
    field_size_ha: Annotated[float, Field(strict=True, ge=0.2, le=4.0)]
    season: Literal["A", "B"]
    cooperative_support: Annotated[int, Field(strict=True, ge=0, le=1)]


class PredictionResponse(BaseModel):
    verification_code: str
    predicted_yield_tons: float
    risk_prediction: str
    high_risk_flag: int
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    human_review_required: bool
    notice: str


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a compact, readable 422 instead of FastAPI's verbose default."""
    details = [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body") or "body", "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details})


@app.get("/health")
def health() -> dict:
    predictor = get_predictor()
    return {"status": "ok", "models_loaded": True, "verification_code": predictor.verification_code}


@app.get("/model-info")
def model_info() -> dict:
    return get_predictor().info()


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord) -> dict:
    predictor = get_predictor()
    try:
        return predictor.predict(record.model_dump())
    except RecordValidationError as exc:  # defence in depth; pydantic should catch these first
        raise HTTPException(status_code=422, detail=exc.errors) from exc
