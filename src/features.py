"""Feature engineering — the single source of truth.

Both training and the FastAPI service must import build_features from this
module. Do not reimplement any part of it elsewhere.
"""
import pandas as pd

RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}

FEATURE_COLUMNS = [
    "PU_DO",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "pickup_hour",
    "pickup_dow",
    "is_weekend",
    "is_rush_hour",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature frame from raw trip rows.

    Expects `PULocationID`, `DOLocationID`, `trip_distance`, and
    `tpep_pickup_datetime` columns. Never touches `tpep_dropoff_datetime` or
    anything derived from it — that would leak the training target.
    """
    out = pd.DataFrame(index=df.index)

    pu = df["PULocationID"].astype(int).astype(str)
    do = df["DOLocationID"].astype(int).astype(str)

    out["PU_DO"] = pu + "_" + do
    out["PULocationID"] = pu
    out["DOLocationID"] = do
    out["trip_distance"] = df["trip_distance"].astype(float)

    pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])
    out["pickup_hour"] = pickup_dt.dt.hour.astype(int)
    out["pickup_dow"] = pickup_dt.dt.dayofweek.astype(int)
    out["is_weekend"] = (out["pickup_dow"] >= 5).astype(int)
    out["is_rush_hour"] = out["pickup_hour"].isin(RUSH_HOURS).astype(int)

    return out[FEATURE_COLUMNS]
