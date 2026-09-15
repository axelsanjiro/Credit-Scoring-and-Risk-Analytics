from fastapi.testclient import TestClient

from src.api import app


def test_health_and_prediction():
    client = TestClient(app)
    response = client.post("/predict", json={"features": {"loan_amnt": 10_000}})
    assert client.get("/health").status_code == 200
    assert response.status_code == 200
    assert 0 <= response.json()["probability_default"] <= 1


def test_api_rejects_conditional_features():
    response = TestClient(app).post(
        "/predict", json={"features": {"loan_amnt": 10_000, "int_rate": 12.0}}
    )
    assert response.status_code == 422
