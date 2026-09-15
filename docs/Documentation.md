# Financial Credit Scoring & Risk Analytics Platform

## Project Objective

This project builds a credit-scoring system to estimate Probability of Default (PD) from historical Lending Club data. PD supports credit scores, Expected Loss, and approval-cutoff recommendations.

This portfolio demonstrates an end-to-end data-science workflow with leakage control, class imbalance handling, probability calibration, explainability, and business decisions.

## Current Status

Implemented stages:

1. Exploratory Data Analysis on raw Lending Club data.
2. Data cleaning and target definition in `src/data_pipeline.py`.
3. Leakage-safe temporal splitting and training-only preprocessing in `src/modeling_pipeline.py`.
4. Deterministic financial-ratio features and missingness indicators.
5. Temporal Logistic Regression, LightGBM, XGBoost, and CatBoost benchmarks.
6. Optuna LightGBM tuning, validation-only calibration, and untouched test evaluation.
7. Expected Loss cutoff analysis, FastAPI endpoint, Streamlit simulator, and unit tests.

Implementation status:

- The planned portfolio pipeline is implemented. Production deployment remains out of scope and requires the controls listed in [Current Limitations](#current-limitations).

See [Model_Improvement_Plan.md](Model_Improvement_Plan.md) for full roadmap.

## Current Pipeline

```text
Raw Lending Club data
    |
    v
Data cleaning and target mapping
    |
    v
Approved deterministic feature engineering
    |
    v
Training-only imputation and encoding after chronological split
    |
    v
Model benchmarking and calibration
    |
    v
ROC-AUC, Gini, KS, PR-AUC, Brier score, and risk deciles
    |
    v
Serialized pipeline, calibrator, feature schema, and decision artifacts
```

## Data And Target Definition

The current ETL reads audited Lending Club fields intended to be available at application or origination. The complete allowed/conditional/excluded feature contract is maintained in [feature_inventory.md](feature_inventory.md).

- Current baseline fields: `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `revol_util`, `delinq_2yrs`, `inq_last_6mths`, `emp_length`, `home_ownership`, and `purpose`
- Expanded approved candidates: loan terms, verification, credit-history, revolving-credit, account-composition, and aggregate-balance fields
- `issue_d` for chronological splitting only
- `loan_status` for target construction only

The target is defined as:

- `Fully Paid` -> `0` (good loan)
- `Charged Off` and `Default` -> `1` (default / bad loan)
- `Current` and all other statuses are excluded because their final repayment outcome is not yet known.

Known post-origination leakage fields, such as repayment, recovery, and collection information, are not part of the ETL whitelist. Examples include `total_pymnt`, `total_rec_prncp`, `recoveries`, and `out_prncp`.

## Current Data Preparation

Implemented in `src/data_pipeline.py`:

- Reads only whitelisted columns to reduce memory use and avoid known leakage fields.
- Converts `int_rate` from percentage text into float.
- Extracts numeric employment duration from `emp_length` into `emp_length_years`.
- Parses `term` into `term_months` and `emp_length` into `emp_length_years`.
- Parses `issue_d` into `issue_date` and derives `credit_history_months` from `earliest_cr_line`.
- Renames `purpose` to `loan_purpose`.
- Retains numeric missing values for training-only imputation in the next pipeline stage.
- Exports `data/processed/credit_applications_cleaned.csv`.

Legacy selected-data and LightGBM artifacts remain available for comparison only.

## Temporal Split And Preprocessing

`src/modeling_pipeline.py` creates chronological partitions by application vintage: oldest 65% for training, next 15% for validation, and newest 20% for final testing. It excludes `issue_date` from predictors and excludes conditional pricing fields (`int_rate`, `installment`) from the default experiment.

The module fits numeric median imputation, numeric missingness indicators, categorical `Missing` treatment, and one-hot encoding on training rows only. Features fully missing in training are excluded from every partition. The fitted preprocessor is saved to `models/credit_preprocessor.pkl`; split metadata is saved to `models/temporal_split_summary.json`.

`src.experiment` fits candidate models on training data, tunes LightGBM with expanding training folds, selects by validation performance, and fits calibration on validation data only. Test rows remain untouched until final evaluation.

## Final Model And Decision Layer

`src.experiment` writes benchmark metrics, Optuna trials, feature importance, risk deciles, cutoff analysis, and `models/credit_risk_model.pkl`. The artifact contains preprocessing, selected estimator, and calibration.

The selected model, calibration method, and metrics are written to `models/final_model_report.json` after each experiment run. Calibration methods are selected using a later chronological portion of the validation period, then refit on the complete validation period. The final test period remains untouched until this step.

Business policy uses LGD `45%`, annual margin `7%`, `$100` operational cost per approval, and a maximum approved default rate of `15%`. The recommended PD cutoff and expected approval rate are recorded in the generated final model report.

Global SHAP review identifies `term_months`, `acc_open_past_24mths`, and `loan_to_income` as top drivers. Run API with `uvicorn src.api:app --reload`. Run simulator with `streamlit run src/dashboard.py`. Build container with `docker build -t credit-risk-api .`.

## Current Limitations

- Final test performance declines from validation performance, indicating temporal drift.
- `term_months` has unusually high XGBoost importance. Keep monitoring decision-point availability and policy dependence.
- Model calibration and business results depend on historical Lending Club outcomes and stated financial assumptions.
- API accepts approved feature names only, but production deployment still needs authentication, audit logging, data contracts, and policy review.
- The supplied raw dataset does not contain FICO-range fields. This limitation is documented in `feature_inventory.md`; alternative credit-profile features will be evaluated instead.
- A `0.5` classification threshold is shown only for diagnostics; it is not a financially optimized loan approval policy.

## Pipeline Design

```text
Raw Lending Club data + data dictionary audit
    |
    v
Approved pre-origination feature set and leakage policy
    |
    v
Out-of-time train / validation / test split
    |
    v
Training-only preprocessing and feature engineering pipeline
    |
    v
Logistic Regression, LightGBM, XGBoost, CatBoost benchmarks
    |
    v
Optuna tuning and validation-period calibration comparison
    |
    v
Untouched out-of-time test evaluation
    |
    v
PD, credit score, Expected Loss, and profit-aware cutoff analysis
    |
    v
Explainable API and interactive dashboard
```

## Modeling Direction

Deep learning is not the primary next step. This is structured tabular data with a limited feature set, where gradient-boosted trees are generally more competitive, efficient, and explainable than neural networks.

The immediate priority is to add valid pre-origination signals, correct the validation design, and tune/compare tabular baselines. A neural network may be added later as an optional benchmark, or if the project adds unstructured data such as documents, text, or transaction sequences.

## Evaluation Framework

The final model will be assessed across three dimensions:

- Discrimination: ROC-AUC, Gini coefficient, KS statistic, and PR-AUC.
- Calibration: Brier score, calibration curve, and observed default rate by PD decile.
- Business impact: approval rate, default rate among approved loans, Expected Loss, and expected profit across possible cutoffs.

Expected Loss is defined as:

```text
Expected Loss = PD x LGD x EAD
```

where PD is the calibrated predicted probability of default, LGD is Loss Given Default, and EAD is Exposure at Default.

## Project Structure

```text
data/
  raw/loan.csv                         # Local, not committed
  processed/credit_applications_cleaned.csv # Generated, not committed
docs/
  Documentation.md
  Model_Improvement_Plan.md
  Model_Card.md
  feature_inventory.md
models/
  credit_risk_model.pkl                # Generated, not committed
  final_model_report.json              # Generated, not committed
notebooks/
  eda_analysis.ipynb
src/
  data_pipeline.py
  modeling_pipeline.py
    experiment.py
    api.py
    dashboard.py
    scoring.py
tests/
  test_api.py
  test_modeling_pipeline.py
```

## Disclaimer

This project uses historical Lending Club data as an educational benchmark. It is not a production lending policy and must not be used to make real credit decisions without governance, legal review, fairness assessment, monitoring, and validation on the intended population.
