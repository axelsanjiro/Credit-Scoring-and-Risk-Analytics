# Credit Scoring & Risk Analytics

An educational, end-to-end credit-risk project using historical Lending Club loans. It estimates Probability of Default (PD), converts PD to a FICO-like score, and evaluates an approval cutoff using Expected Loss and profit assumptions.

This repository is a portfolio project, not a production lending policy. Do not use it to make real credit decisions.

## What It Includes

- Leakage-audited Lending Club ETL and target construction.
- Chronological train, validation, and final test partitions.
- Training-only imputation, missing-value indicators, and categorical encoding.
- Logistic Regression, LightGBM, XGBoost, and CatBoost benchmarks.
- Optuna tuning, calibration selection, risk deciles, feature importance, and SHAP summaries.
- FastAPI inference endpoint and Streamlit scoring simulator.
- Unit tests for temporal splitting, financial ratios, score scaling, and API validation.

## Repository Layout

```text
data/
  raw/                 # Local source data, not committed
  processed/           # Generated datasets, not committed
docs/
  Documentation.md     # Technical pipeline documentation
  Model_Card.md        # Intended use, results, risks, and monitoring
  feature_inventory.md # Feature availability and leakage policy
models/                # Generated model and experiment artifacts, not committed
notebooks/
  eda_analysis.ipynb   # Exploratory analysis
src/
  data_pipeline.py     # ETL and target definition
  modeling_pipeline.py # Temporal split and preprocessing
  experiment.py        # Benchmarking, calibration, and cutoff analysis
  api.py               # FastAPI application
  dashboard.py         # Streamlit application
  scoring.py           # PD-to-credit-score conversion
tests/                 # Automated tests
```

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Download the [Lending Club Loan Data CSV](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data) from Kaggle. Place the CSV at `data/raw/loan.csv`. Source data and generated artifacts are excluded from Git because of size and licensing considerations.

## Reproduce The Pipeline

Run commands from the repository root:

```bash
python -m src.data_pipeline
python -m src.experiment
pytest -q
```

`src.experiment` writes the deployable model to `models/credit_risk_model.pkl` and stores metrics, calibration diagnostics, deciles, SHAP summaries, and cutoff analysis in `models/`.

## Run Applications

Generate the model artifacts first, then start either application:

```bash
uvicorn src.api:app --reload
streamlit run src/dashboard.py
```

Open Streamlit at [http://localhost:8501](http://localhost:8501). Open FastAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs).

The API exposes `GET /health` and `POST /predict`. Send only fields in the trained feature schema; `loan_amnt` is required and must be positive. Conditional pricing fields such as `int_rate` and `installment` are rejected.

## Docker

The image expects generated `models/` artifacts in the build context. Run the experiment before building:

```bash
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 credit-risk-api
```

## Documentation

- [Technical documentation](docs/Documentation.md)
- [Model card](docs/Model_Card.md)
- [Feature inventory and leakage policy](docs/feature_inventory.md)
- [Improvement plan](docs/Model_Improvement_Plan.md)

## Limitations

- Results are specific to historical Lending Club data and may drift over time.
- The approval policy depends on documented LGD, margin, and operating-cost assumptions.
- A production system requires data contracts, authentication, audit logging, fairness assessment, monitoring, and policy approval.
