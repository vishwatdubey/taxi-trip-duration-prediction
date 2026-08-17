"""FastAPI service for the NYC taxi trip duration model."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_features

MLFLOW_TRACKING_URI = "file:./mlruns"
REGISTERED_MODEL_NAME = "nyc-taxi-duration"
CHAMPION_PATH = Path("models/champion.json")

model_state = {"model": None, "version": None}


def _load_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        return model, str(mv.version)
    except Exception:
        with open(CHAMPION_PATH) as f:
            champion = json.load(f)
        model = mlflow.sklearn.load_model(f"runs:/{champion['run_id']}/model")
        return model, str(champion["version"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, version = _load_champion_model()
        model_state["model"] = model
        model_state["version"] = version
    except Exception as exc:  # noqa: BLE001
        model_state["model"] = None
        model_state["version"] = None
        model_state["load_error"] = str(exc)
    yield
    model_state["model"] = None


app = FastAPI(title="NYC Taxi Trip Duration", lifespan=lifespan)


class TripRequest(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float = Field(gt=0)
    passenger_count: int = Field(ge=1, le=6)
    pickup_datetime: str


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_version: str


def _require_model():
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    return model_state["model"]


def _to_frame(requests: list[TripRequest]) -> pd.DataFrame:
    rows = []
    for r in requests:
        rows.append(
            {
                "PULocationID": r.PULocationID,
                "DOLocationID": r.DOLocationID,
                "trip_distance": r.trip_distance,
                "passenger_count": r.passenger_count,
                "tpep_pickup_datetime": pd.Timestamp(r.pickup_datetime),
            }
        )
    return pd.DataFrame(rows)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TripRequest):
    model = _require_model()
    df = _to_frame([request])
    features = build_features(df)
    pred = float(model.predict(features)[0])
    return PredictionResponse(
        predicted_duration_minutes=round(pred, 2),
        model_version=model_state["version"],
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: list[TripRequest]):
    model = _require_model()
    df = _to_frame(requests)
    features = build_features(df)
    preds = model.predict(features)
    return [
        PredictionResponse(
            predicted_duration_minutes=round(float(p), 2),
            model_version=model_state["version"],
        )
        for p in preds
    ]
