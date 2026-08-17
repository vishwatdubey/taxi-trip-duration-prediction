import pandas as pd

from src.train import clean


def _base_row(**overrides):
    row = {
        "tpep_pickup_datetime": pd.Timestamp("2023-01-15T08:00:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2023-01-15T08:10:00"),  # 10 min
        "trip_distance": 2.0,
    }
    row.update(overrides)
    return row


def test_duration_outlier_bounds_filter():
    df = pd.DataFrame(
        [
            _base_row(),  # 10 min, distance 2 -> kept
            _base_row(
                tpep_dropoff_datetime=pd.Timestamp("2023-01-15T08:00:30")
            ),  # 0.5 min -> dropped (< 1 min)
            _base_row(
                tpep_dropoff_datetime=pd.Timestamp("2023-01-15T09:30:00")
            ),  # 90 min -> dropped (> 60 min)
        ]
    )
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["duration_min"] == 10


def test_trip_distance_bounds_filter():
    df = pd.DataFrame(
        [
            _base_row(trip_distance=2.0),  # kept
            _base_row(trip_distance=0.0),  # dropped (<= 0)
            _base_row(trip_distance=150.0),  # dropped (> 100)
        ]
    )
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["trip_distance"] == 2.0


def test_implied_speed_filter():
    df = pd.DataFrame(
        [
            _base_row(trip_distance=2.0),  # 12 mph -> kept
            _base_row(
                trip_distance=50.0,
                tpep_dropoff_datetime=pd.Timestamp("2023-01-15T08:01:00"),
            ),  # 50 miles in 1 min -> 3000 mph, dropped
        ]
    )
    out = clean(df)
    assert len(out) == 1
