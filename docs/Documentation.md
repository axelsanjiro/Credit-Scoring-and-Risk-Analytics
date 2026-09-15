# Financial Credit Scoring & Risk Analytics Platform

## Project Objective

Project ini membangun sistem credit scoring untuk mengestimasi Probability of Default (PD) pada data historis Lending Club. Output PD akan menjadi dasar untuk credit score, Expected Loss, dan rekomendasi approval cutoff.

Tujuan portofolio proyek adalah menunjukkan workflow data science end-to-end yang memperhatikan data leakage, class imbalance, probability calibration, explainability, dan keputusan bisnis.

## Current Status

Tahap yang sudah diimplementasikan:

1. Exploratory Data Analysis pada raw data Lending Club.
2. Data cleaning dan target definition di `src/data_pipeline.py`.
3. Information Value (IV) screening di `src/feature_engineering.py`.
4. LightGBM training, isotonic calibration, dan holdout evaluation di `src/train.py`.

Tahap yang belum diimplementasikan:

- Out-of-time validation.
- Optuna hyperparameter tuning.
- Benchmark XGBoost, CatBoost, dan Logistic Regression + WoE.
- Financial ratio engineering yang tervalidasi.
- Business cutoff / Expected Loss optimizer.
- FastAPI, Streamlit, Docker, SHAP serving, dan automated tests.

Roadmap rinci tersedia di [Model_Improvement_Plan.md](Model_Improvement_Plan.md).

## Current Pipeline

```text
Raw Lending Club data
    |
    v
Data cleaning and target mapping
    |
    v
IV-based feature screening
    |
    v
One-hot encoding and stratified random 80:20 split
    |
    v
LightGBM + isotonic calibration
    |
    v
ROC-AUC, Gini, KS, Brier score, classification report
    |
    v
Serialized model and feature names
```

## Data And Target Definition

The current ETL reads a restricted set of Lending Club columns that are intended to be available before or at loan origination:

- `loan_amnt`
- `int_rate`
- `annual_inc`
- `dti`
- `fico_range_low`
- `revol_util`
- `delinq_2yrs`
- `inq_last_6mths`
- `emp_length`
- `home_ownership`
- `purpose`
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
- Median-imputes `annual_inc`, `dti`, `revol_util`, and `int_rate`.
- Fills selected credit-count fields with zero.
- Renames `fico_range_low` to `fico_score` and `purpose` to `loan_purpose` when those source columns are present.
- Drops remaining missing rows and exports `data/processed/credit_applications_cleaned.csv`.

The generated cleaned artifact currently contains 1,303,638 rows, 10 predictor columns, and a default rate of about 20.07%.

## Current Feature Screening

`src/feature_engineering.py` calculates Weight of Evidence (WoE) and Information Value (IV) using quantile bins for continuous variables. Features with IV >= `0.02` are exported to `data/processed/credit_applications_selected.csv`.

The current selected artifact retains these seven raw predictors:

- `loan_amnt`
- `int_rate`
- `home_ownership`
- `annual_inc`
- `dti`
- `inq_last_6mths`
- `revol_util`

Current IV screening results show that `int_rate` is the strongest individual signal (IV about `0.4468`). Most remaining retained variables have weak IV. This supports the hypothesis that the current performance ceiling is caused mainly by limited feature signal.

WoE/IV is useful for diagnostics and an interpretable scorecard baseline. It will not be used as the only hard filter for tree models in the improved pipeline because low univariate IV features can still add predictive value through interactions.

## Current Model

`src/train.py` performs the following steps:

1. Loads the selected dataset, or falls back to the cleaned dataset.
2. One-hot encodes categorical fields using `pandas.get_dummies`.
3. Creates a stratified random 80:20 train-test split with `random_state=42`.
4. Calculates `scale_pos_weight` from the training class ratio.
5. Trains an `LGBMClassifier` with fixed hyperparameters.
6. Applies isotonic calibration using `CalibratedClassifierCV` with 3 folds.
7. Evaluates holdout ROC-AUC, Gini, KS statistic, Brier score, and the classification report at a `0.5` threshold.
8. Saves the calibrated model to `models/credit_lgbm_model.pkl` and model feature names to `models/model_features.pkl`.

## Current Baseline Result

Reconstructed evaluation of the existing model artifact using its deterministic holdout split produced:

| Metric | Result | Reference Target |
|---|---:|---:|
| ROC-AUC | 0.7025 | > 0.75 |
| Gini coefficient | 0.4050 | > 0.50 |
| KS statistic | 0.2924 | > 0.40 |
| Brier score | 0.1468 | Lower is better |

The current baseline does not meet the intended discrimination targets. These values should be treated as a baseline, not as a production-ready credit model.

## Current Limitations

- The dataset is randomly split rather than split by application vintage; future stability is not yet tested.
- Imputation, IV calculation, and one-hot encoding occur before the split. This creates methodological leakage because test-distribution information influences preprocessing and selection.
- The model uses fixed LightGBM parameters. Optuna tuning and model comparison are not implemented yet.
- Metrics are printed to the console but are not persisted as experiment artifacts.
- The preprocessing steps are not serialized with the model. An API cannot yet safely transform raw applicant input into the exact model schema.
- The documentation and generated processed artifact need a data audit to resolve the current absence of `fico_score` from the artifact.
- A `0.5` classification threshold is shown only for diagnostics; it is not a financially optimized loan approval policy.

## Planned Improved Pipeline

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
Logistic Regression + WoE, LightGBM, XGBoost, CatBoost benchmarks
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
  raw/loan.csv
  processed/credit_applications_cleaned.csv
  processed/credit_applications_selected.csv
docs/
  Documentation.md
  Model_Improvement_Plan.md
models/
  credit_lgbm_model.pkl
  model_features.pkl
notebooks/
  eda_analysis.ipynb
src/
  data_pipeline.py
  feature_engineering.py
  train.py
```

## Disclaimer

This project uses historical Lending Club data as an educational benchmark. It is not a production lending policy and must not be used to make real credit decisions without governance, legal review, fairness assessment, monitoring, and validation on the intended population.
