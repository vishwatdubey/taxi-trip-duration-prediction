"""Phase 1 + 2: clean data, train three models, log to MLflow, register champion."""
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features import build_features

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "nyc-taxi-duration"
REGISTERED_MODEL_NAME = "nyc-taxi-duration"
SAMPLE_PATH = Path("data/processed/sample.parquet")
CHAMPION_PATH = Path("models/champion.json")


class DictVectorizerTransformer(BaseEstimator, TransformerMixin):
    """Wraps a fitted DictVectorizer as a sklearn-pipeline-compatible transformer."""

    def __init__(self, dv: DictVectorizer):
        self.dv = dv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.dv.transform(X.to_dict(orient="records"))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 60)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]

    implied_speed_mph = df["trip_distance"] / (df["duration_min"] / 60)
    df = df[implied_speed_mph <= 100]

    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_pipeline(dv: DictVectorizer, estimator) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizerTransformer(dv)),
            ("model", estimator),
        ]
    )


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(SAMPLE_PATH)
    df = clean(df)
    train_df, val_df = temporal_split(df)

    assert train_df["tpep_pickup_datetime"].max() < val_df["tpep_pickup_datetime"].min(), (
        "temporal split violated"
    )

    X_train_raw = build_features(train_df)
    X_val_raw = build_features(val_df)
    y_train = train_df["duration_min"].values
    y_val = val_df["duration_min"].values

    dv = DictVectorizer(sparse=True)
    dv.fit(X_train_raw.to_dict(orient="records"))

    models = {
        "linear_regression": (LinearRegression(), {}),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=50, max_depth=12, n_jobs=-1, random_state=42
            ),
            {"n_estimators": 50, "max_depth": 12},
        ),
        "xgboost": (
            XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
            ),
            {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1},
        ),
    }

    results = {}
    run_ids = {}

    for name, (estimator, params) in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = build_pipeline(dv, estimator)
            pipeline.fit(X_train_raw, y_train)
            preds = pipeline.predict(X_val_raw)
            metrics = compute_metrics(y_val, preds)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            dv_path = Path("models") / f"dv_{name}.json"
            dv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dv_path, "w") as f:
                json.dump(dv.vocabulary_, f)
            mlflow.log_artifact(str(dv_path), artifact_path="vectorizer")

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            results[name] = metrics
            run_ids[name] = run.info.run_id

    print("\n=== Metrics (validation split, minutes) ===")
    print(f"{'model':<20}{'rmse':>10}{'mae':>10}{'r2':>10}")
    for name, m in results.items():
        print(f"{name:<20}{m['rmse']:>10.3f}{m['mae']:>10.3f}{m['r2']:>10.4f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_run_id = run_ids[best_name]
    print(f"\nBest model: {best_name} (run_id={best_run_id}, r2={results[best_name]['r2']:.4f})")

    client = mlflow.MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)

    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_PATH, "w") as f:
        json.dump(
            {
                "run_id": best_run_id,
                "model_name": best_name,
                "version": mv.version,
                "r2": results[best_name]["r2"],
            },
            f,
            indent=2,
        )

    xgb_r2 = results["xgboost"]["r2"]
    print(f"\nXGBoost R2 = {xgb_r2:.4f} (acceptance threshold: >= 0.65)")
    assert xgb_r2 >= 0.65, f"XGBoost R2 {xgb_r2:.4f} below acceptance threshold 0.65"


if __name__ == "__main__":
    main()
