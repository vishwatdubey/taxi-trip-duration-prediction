from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "PULocationID": 142,
    "DOLocationID": 236,
    "trip_distance": 3.4,
    "passenger_count": 1,
    "pickup_datetime": "2023-01-15T08:30:00",
}


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_valid():
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["predicted_duration_minutes"], (int, float))
        assert 0 <= body["predicted_duration_minutes"] <= 120
        assert "model_version" in body


def test_predict_malformed_returns_422():
    with TestClient(app) as client:
        bad_payload = dict(VALID_PAYLOAD)
        bad_payload["trip_distance"] = "not-a-number"
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422


def test_predict_batch():
    with TestClient(app) as client:
        resp = client.post("/predict/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        for item in body:
            assert 0 <= item["predicted_duration_minutes"] <= 120
