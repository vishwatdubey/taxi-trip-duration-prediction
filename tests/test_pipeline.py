"""The logged pipeline must accept raw-shaped feature input directly, i.e.
the DictVectorizer travels with the model inside the pipeline."""
import pandas as pd
import pytest

from scripts.load_champion import load_champion
from src.features import build_features


@pytest.fixture(scope="module")
def champion_model():
    return load_champion()


def test_pipeline_accepts_raw_feature_dataframe(champion_model):
    df = pd.DataFrame(
        [
            {
                "PULocationID": 142,
                "DOLocationID": 236,
                "trip_distance": 3.4,
                "passenger_count": 1,
                "tpep_pickup_datetime": pd.Timestamp("2023-01-15T08:30:00"),
            }
        ]
    )
    features = build_features(df)
    # No manual DictVectorizer call here: the pipeline must vectorize internally.
    pred = champion_model.predict(features)
    assert pred.shape == (1,)
    assert 0 <= pred[0] <= 180


def test_pipeline_has_vectorizer_step(champion_model):
    step_names = [name for name, _ in champion_model.steps]
    assert "vectorizer" in step_names
    assert "model" in step_names
