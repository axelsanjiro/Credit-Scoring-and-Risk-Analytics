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

## Evaluation Results

Latest experiment selects XGBoost with isotonic calibration. Final metrics use untouched out-of-time test data from September 2016 through December 2018.

| Metric | Validation | Final Test |
|---|---:|---:|
| ROC-AUC | 0.7249 | 0.6968 |
| Gini | 0.4499 | 0.3936 |
| KS | 0.3240 | 0.2853 |
| PR-AUC | 0.4224 | 0.3730 |
| Brier score | 0.1519 | 0.1575 |

Recommended PD cutoff: `0.23`. Expected approval rate: `64.61%`. Approved-loan default rate: `14.69%`. Results depend on historical data and documented business assumptions.

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

The API exposes `GET /health` and `POST /predict`. It requires core application, affordability, and credit-history fields; additional supported bureau fields are optional. Derived ratios are calculated by the API. Conditional pricing fields such as `int_rate` and `installment` are rejected.

### API Prediction Example

Start the API, then send the required application and bureau fields. Additional supported fields are optional and missing optional values are handled by the training-time preprocessing pipeline.

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":{"loan_amnt":10000,"annual_inc":75000,"dti":18.5,"delinq_2yrs":0,"inq_last_6mths":1,"revol_bal":12000,"revol_util":42.0,"total_acc":18,"total_rev_hi_lim":30000,"credit_history_months":144,"term_months":36,"emp_length_years":5,"home_ownership":"RENT","verification_status":"Verified","loan_purpose":"debt_consolidation"}}'
```

The response contains `probability_default`, a FICO-like `credit_score`, the selected `pd_cutoff`, an `approved` decision, and `expected_loss`.

```json
{
  "probability_default": 0.18,
  "credit_score": 658,
  "pd_cutoff": 0.23,
  "approved": true,
  "expected_loss": 810.0
}
```

Values above are illustrative; predictions vary with the request and generated model artifact. Use [http://localhost:8000/docs](http://localhost:8000/docs) to inspect and try the interactive OpenAPI schema.

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
