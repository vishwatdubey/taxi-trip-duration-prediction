import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def _sample_df():
    return pd.DataFrame(
        {
            "PULocationID": [142, 236],
            "DOLocationID": [236, 142],
            "trip_distance": [3.4, 1.1],
            "tpep_pickup_datetime": [
                pd.Timestamp("2023-01-15T08:30:00"),  # Sunday, rush hour
                pd.Timestamp("2023-01-17T14:00:00"),  # Tuesday, not rush hour
            ],
            "tpep_dropoff_datetime": [
                pd.Timestamp("2023-01-15T08:45:00"),
                pd.Timestamp("2023-01-17T14:10:00"),
            ],
        }
    )


def test_output_columns_exact():
    out = build_features(_sample_df())
    assert list(out.columns) == FEATURE_COLUMNS


def test_no_dropoff_derived_column_present():
    out = build_features(_sample_df())
    for col in out.columns:
        assert "dropoff" not in col.lower()


def test_build_features_is_deterministic():
    df = _sample_df()
    out1 = build_features(df)
    out2 = build_features(df)
    pd.testing.assert_frame_equal(out1, out2)


def test_pu_do_concat():
    out = build_features(_sample_df())
    assert out.loc[0, "PU_DO"] == "142_236"


def test_rush_hour_and_weekend_flags():
    out = build_features(_sample_df())
    assert out.loc[0, "pickup_hour"] == 8
    assert out.loc[0, "is_rush_hour"] == 1
    assert out.loc[1, "is_rush_hour"] == 0
