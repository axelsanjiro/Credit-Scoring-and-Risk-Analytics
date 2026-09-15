"""FastAPI inference endpoint for approved pre-origination loan fields."""

import os
import json
from typing import Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from src.scoring import prob_to_credit_score


ARTIFACT_PATH = "models/credit_risk_model.pkl"
REPORT_PATH = "models/final_model_report.json"
Value = Union[float, int, str, None]

if not os.path.exists(ARTIFACT_PATH):
    raise RuntimeError(f"Model artifact not found: {ARTIFACT_PATH}. Run src.experiment first.")

MODEL = joblib.load(ARTIFACT_PATH)
PREPROCESSOR = MODEL.pipeline.named_steps["preprocessor"]
NUMERIC_FEATURES = list(PREPROCESSOR.transformers_[0][2])
CATEGORICAL_FEATURES = list(PREPROCESSOR.transformers_[1][2])
ALLOWED_FEATURES = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
with open(REPORT_PATH, encoding="ascii") as file:
    PD_CUTOFF = json.load(file)["recommended_cutoff"]["pd_cutoff"]


class PredictionRequest(BaseModel):
    features: dict[str, Value]

    @model_validator(mode="after")
    def check_features(self):
        unexpected = sorted(set(self.features) - ALLOWED_FEATURES)
        if unexpected:
            raise ValueError(f"Unsupported or conditional features: {unexpected}")
        loan_amount = self.features.get("loan_amnt")
        if not isinstance(loan_amount, (int, float)) or loan_amount <= 0:
            raise ValueError("'loan_amnt' must be a positive number.")
        return self


app = FastAPI(title="Credit Risk API", version="1.0.0")


def to_model_frame(features: dict[str, Value]) -> pd.DataFrame:
    row = {feature: features.get(feature) for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    frame = pd.DataFrame([row])
    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "feature_count": len(ALLOWED_FEATURES)}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        frame = to_model_frame(request.features)
        probability_default = float(MODEL.predict_proba(frame)[0, 1])
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    loan_amount = float(frame.loc[0, "loan_amnt"])
    expected_loss = probability_default * 0.45 * loan_amount
    return {
        "probability_default": probability_default,
        "credit_score": int(prob_to_credit_score(pd.Series([probability_default]).to_numpy())[0]),
        "pd_cutoff": PD_CUTOFF,
        "approved": probability_default < PD_CUTOFF,
        "expected_loss": expected_loss,
    }
