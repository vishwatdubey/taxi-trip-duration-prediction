import numpy as np
import pandas as pd

from src.train import temporal_split


def test_train_max_before_val_min():
    n = 1000
    rng = np.random.default_rng(0)
    timestamps = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.permutation(n), unit="m"
    )
    df = pd.DataFrame({"tpep_pickup_datetime": timestamps, "value": range(n)})

    train_df, val_df = temporal_split(df)

    assert len(train_df) + len(val_df) == n
    assert train_df["tpep_pickup_datetime"].max() < val_df["tpep_pickup_datetime"].min()


def test_split_ratio_is_80_20():
    n = 1000
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=n, freq="min"),
            "value": range(n),
        }
    )
    train_df, val_df = temporal_split(df)
    assert len(train_df) == 800
    assert len(val_df) == 200
