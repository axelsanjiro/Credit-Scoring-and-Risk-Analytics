"""Streamlit simulator for the persisted credit-risk model."""

import json
import os

import joblib
import pandas as pd
import streamlit as st

from src.api import CATEGORICAL_FEATURES, NUMERIC_FEATURES, to_model_frame
from src.scoring import prob_to_credit_score


st.set_page_config(page_title="Credit Risk Simulator", layout="wide")
st.title("Credit Risk Simulator")
st.caption("Historical Lending Club benchmark. Not production lending policy.")

model = joblib.load("models/credit_risk_model.pkl")
report_path = "models/final_model_report.json"
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

if os.path.exists("models/shap_summary.csv"):
    st.subheader("Top global SHAP drivers")
    st.bar_chart(pd.read_csv("models/shap_summary.csv").head(10), x="feature", y="mean_absolute_shap")

if os.path.exists("models/cutoff_analysis.csv"):
    st.subheader("Cutoff trade-off")
    st.line_chart(pd.read_csv("models/cutoff_analysis.csv"), x="pd_cutoff", y="expected_profit")
