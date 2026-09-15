"""FastAPI inference endpoint for approved pre-origination loan fields."""

import json
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.modeling_pipeline import add_financial_features
from src.scoring import prob_to_credit_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "credit_risk_model.pkl"
REPORT_PATH = PROJECT_ROOT / "models" / "final_model_report.json"
Value = Union[float, int, str, None]

if not ARTIFACT_PATH.exists():
    raise RuntimeError(f"Model artifact not found: {ARTIFACT_PATH}. Run src.experiment first.")

MODEL = joblib.load(ARTIFACT_PATH)
PREPROCESSOR = MODEL.pipeline.named_steps["preprocessor"]
NUMERIC_FEATURES = list(PREPROCESSOR.transformers_[0][2])
CATEGORICAL_FEATURES = list(PREPROCESSOR.transformers_[1][2])
ALLOWED_FEATURES = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
DERIVED_FEATURES = {
    "loan_to_income",
    "revolving_balance_to_income",
    "revolving_balance_to_limit",
    "accounts_per_credit_history_year",
}
REQUEST_FEATURES = ALLOWED_FEATURES - DERIVED_FEATURES
REQUIRED_FEATURES = {
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_bal",
    "revol_util",
    "total_acc",
    "total_rev_hi_lim",
    "credit_history_months",
    "term_months",
    "emp_length_years",
    "home_ownership",
    "verification_status",
    "loan_purpose",
}
POSITIVE_FEATURES = {"loan_amnt", "annual_inc", "total_rev_hi_lim", "credit_history_months", "term_months"}
NON_NEGATIVE_FEATURES = REQUIRED_FEATURES - {
    "home_ownership", "verification_status", "loan_purpose"
}
with open(REPORT_PATH, encoding="ascii") as file:
    PD_CUTOFF = json.load(file)["recommended_cutoff"]["pd_cutoff"]


class PredictionRequest(BaseModel):
    features: dict[str, Value] = Field(
        json_schema_extra={
            "example": {
                "loan_amnt": 10000,
                "annual_inc": 75000,
                "dti": 18.5,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "revol_bal": 12000,
                "revol_util": 42.0,
                "total_acc": 18,
                "total_rev_hi_lim": 30000,
                "credit_history_months": 144,
                "term_months": 36,
                "emp_length_years": 5,
                "home_ownership": "RENT",
                "verification_status": "Verified",
                "loan_purpose": "debt_consolidation",
            }
        }
    )

    @model_validator(mode="after")
    def check_features(self):
        unexpected = sorted(set(self.features) - REQUEST_FEATURES)
        if unexpected:
            raise ValueError(f"Unsupported, derived, or conditional features: {unexpected}")
        missing = sorted(REQUIRED_FEATURES - set(self.features))
        if missing:
            raise ValueError(f"Missing required application features: {missing}")
        for feature in NON_NEGATIVE_FEATURES:
            value = self.features[feature]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"'{feature}' must be a non-negative number.")
        for feature in POSITIVE_FEATURES:
            if self.features[feature] <= 0:
                raise ValueError(f"'{feature}' must be a positive number.")
        for feature in {"home_ownership", "verification_status", "loan_purpose"}:
            if not isinstance(self.features[feature], str) or not self.features[feature].strip():
                raise ValueError(f"'{feature}' must be a non-empty string.")
        return self


class PredictionResponse(BaseModel):
    probability_default: float
    credit_score: int
    pd_cutoff: float
    approved: bool
    expected_loss: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "probability_default": 0.18,
                "credit_score": 658,
                "pd_cutoff": 0.23,
                "approved": True,
                "expected_loss": 810.0,
            }
        }
    }


app = FastAPI(title="Credit Risk API", version="1.0.0")


def to_model_frame(features: dict[str, Value]) -> pd.DataFrame:
    row = {feature: features.get(feature) for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    frame = pd.DataFrame([row])
    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return add_financial_features(frame)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "feature_count": len(ALLOWED_FEATURES)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        frame = to_model_frame(request.features)
        probability_default = float(MODEL.predict_proba(frame)[0, 1])
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    loan_amount = float(frame.loc[0, "loan_amnt"])
    expected_loss = probability_default * 0.45 * loan_amount
    return PredictionResponse(
        probability_default=probability_default,
        credit_score=int(prob_to_credit_score(pd.Series([probability_default]).to_numpy())[0]),
        pd_cutoff=PD_CUTOFF,
        approved=probability_default < PD_CUTOFF,
        expected_loss=expected_loss,
    )
