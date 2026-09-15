from fastapi.testclient import TestClient

from src.api import app


VALID_FEATURES = {
    "loan_amnt": 10_000,
    "annual_inc": 75_000,
    "dti": 18.5,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "revol_bal": 12_000,
    "revol_util": 42.0,
    "total_acc": 18,
    "total_rev_hi_lim": 30_000,
    "credit_history_months": 144,
    "term_months": 36,
    "emp_length_years": 5,
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "loan_purpose": "debt_consolidation",
}


def test_health_and_prediction():
    client = TestClient(app)
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert client.get("/health").status_code == 200
    assert response.status_code == 200
    assert 0 <= response.json()["probability_default"] <= 1


def test_api_rejects_conditional_features():
    response = TestClient(app).post(
        "/predict", json={"features": {**VALID_FEATURES, "int_rate": 12.0}}
    )
    assert response.status_code == 422
