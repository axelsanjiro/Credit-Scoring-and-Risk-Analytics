"""Streamlit simulator for the persisted credit-risk model."""

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Streamlit executes this file directly, so add the project root for `src` imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import CATEGORICAL_FEATURES, NUMERIC_FEATURES, to_model_frame
from src.scoring import prob_to_credit_score


st.set_page_config(page_title="Credit Risk Simulator", layout="wide")
st.title("Credit Risk Simulator")
st.caption("Historical Lending Club benchmark. Not production lending policy.")

model = joblib.load(PROJECT_ROOT / "models" / "credit_risk_model.pkl")
report_path = PROJECT_ROOT / "models" / "final_model_report.json"
report = json.load(open(report_path, encoding="ascii")) if os.path.exists(report_path) else {}
cutoff = report.get("recommended_cutoff", {}).get("pd_cutoff", 0.23)

with st.form("application"):
    values = {}
    left, right = st.columns(2)
    for index, feature in enumerate(NUMERIC_FEATURES):
        container = left if index % 2 == 0 else right
        values[feature] = container.number_input(feature, value=0.0)
    for index, feature in enumerate(CATEGORICAL_FEATURES):
        values[feature] = st.text_input(feature)
    submitted = st.form_submit_button("Score application")

if submitted:
    frame = to_model_frame(values)
    probability_default = float(model.predict_proba(frame)[0, 1])
    st.metric("Probability of Default", f"{probability_default:.2%}")
    st.metric("Credit score", int(prob_to_credit_score(pd.Series([probability_default]).to_numpy())[0]))
    st.metric("Expected loss", f"${probability_default * 0.45 * values['loan_amnt']:,.2f}")
    st.write("Decision:", "Approve" if probability_default < cutoff else "Decline")

shap_path = PROJECT_ROOT / "models" / "shap_summary.csv"
if os.path.exists(shap_path):
    st.subheader("Top global SHAP drivers")
    st.bar_chart(pd.read_csv(shap_path).head(10), x="feature", y="mean_absolute_shap")

cutoff_path = PROJECT_ROOT / "models" / "cutoff_analysis.csv"
if os.path.exists(cutoff_path):
    st.subheader("Cutoff trade-off")
    st.line_chart(pd.read_csv(cutoff_path), x="pd_cutoff", y="expected_profit")
