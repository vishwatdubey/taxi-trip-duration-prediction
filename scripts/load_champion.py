"""Phase 2 acceptance check: load the champion model and predict on one input.

Tries the alias URI first (`models:/nyc-taxi-duration@champion`). If alias
loading is flaky with the file store, falls back to the run ID recorded in
models/champion.json (see DECISIONS.md for why this fallback exists).
"""
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.features import build_features

MLFLOW_TRACKING_URI = "file:./mlruns"
REGISTERED_MODEL_NAME = "nyc-taxi-duration"
CHAMPION_PATH = Path("models/champion.json")

SAMPLE_INPUT = {
    "PULocationID": 142,
    "DOLocationID": 236,
    "trip_distance": 3.4,
    "passenger_count": 1,
    "tpep_pickup_datetime": pd.Timestamp("2023-01-15T08:30:00"),
}


def load_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
        print("Loaded champion via alias URI.")
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"Alias loading failed ({exc}); falling back to models/champion.json run_id.")
        with open(CHAMPION_PATH) as f:
            champion = json.load(f)
        model = mlflow.sklearn.load_model(f"runs:/{champion['run_id']}/model")
        print(f"Loaded champion via run_id fallback ({champion['run_id']}).")
        return model


def main() -> None:
    model = load_champion()
    df = pd.DataFrame([SAMPLE_INPUT])
    features = build_features(df)
    pred = model.predict(features)[0]
    print(f"Predicted duration for sample input: {pred:.2f} minutes")


if __name__ == "__main__":
    main()
