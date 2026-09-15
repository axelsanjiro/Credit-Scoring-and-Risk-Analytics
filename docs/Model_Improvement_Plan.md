# Credit Scoring Model Improvement Plan

## Objective

Meningkatkan model Probability of Default (PD) secara valid dan dapat dijelaskan. Fokus utama bukan mengganti LightGBM dengan deep learning, tetapi memperkaya sinyal fitur pre-origination, mencegah data leakage, dan mengevaluasi model seperti sistem credit risk di dunia nyata.

## Current Baseline

| Metric | Current Result | Target |
|---|---:|---:|
| ROC-AUC | 0.7025 | > 0.75 |
| Gini | 0.4050 | > 0.50 |
| KS | 0.2924 | > 0.40 |
| Brier Score | 0.1468 | Lower is better |

Current model uses LightGBM on a small set of selected features. The most predictive feature is `int_rate`; therefore, the main limitation is likely insufficient pre-origination signal rather than the model family or dataset size.

## Scope And Principles

- Use only information available at loan application or approval time.
- Never use repayment, collection, recovery, or other post-origination features.
- Split data before fitting any learned preprocessing, feature selection, or calibration step.
- Keep an out-of-time test set untouched until final model selection.
- Evaluate discrimination, calibration, and business impact separately.
- Prioritize reproducibility, explainability, and deployment consistency.

## Phase 1 - Data And Feature Audit

### Goal

Create an explicit, auditable feature inventory before changing the model.

### Tasks

1. Read the Lending Club data dictionary and classify every candidate column as one of:
   - Safe pre-origination feature.
   - Post-origination leakage feature.
   - Identifier or high-cardinality field requiring special treatment.
   - Unavailable or unclear-at-application feature.
2. Create `docs/feature_inventory.md` with column name, business definition, availability timing, leakage decision, and rationale.
3. Verify why `fico_range_low` / `fico_score` is absent from the current processed artifact despite being listed in `data_pipeline.py` and documentation.
4. Align `Documentation.md`, pipeline code, and generated artifacts so they describe the same model input.

### Candidate Safe Features

Validate against the data dictionary before use:

- Credit profile: `fico_range_low`, `fico_range_high`, `open_acc`, `total_acc`, `mort_acc`, `pub_rec`, `pub_rec_bankruptcies`, `delinq_2yrs`, `inq_last_6mths`.
- Loan application: `loan_amnt`, `term`, `purpose`, `home_ownership`, `verification_status`, `application_type`, `initial_list_status`.
- Affordability: `annual_inc`, `dti`, `installment`, `revol_bal`, `revol_util`.
- Lending Club risk/pricing fields: `grade`, `sub_grade`, `int_rate`.

### Explicit Leakage Exclusions

Do not use fields such as `loan_status`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `out_prncp`, or any feature recorded after loan origination.

### Success Criteria

- Every modeled field has a documented availability decision.
- The processed dataset contains all approved candidate predictors.
- No post-origination field can enter the model pipeline.

## Phase 2 - Leakage-Safe Data Split And Preprocessing

### Goal

Build one reproducible pipeline that learns exclusively from training data.

### Tasks

1. Retain and parse `issue_d` only for splitting, not necessarily as a predictive feature.
2. Reserve the newest period as the final out-of-time test set.
3. Split the remaining historical data into train and validation sets, preserving temporal order.
4. Fit the following components on training data only:
   - Numeric median imputer.
   - Categorical missing-value treatment.
   - Outlier capping where justified.
   - Encoder or native categorical handling.
   - Feature selection.
   - Probability calibrator.
5. Serialize preprocessing and model together as a single artifact.

### Recommended Split

- Training: oldest 60-70% of application vintages.
- Validation: next 10-20% of vintages for tuning and calibration selection.
- Test: newest 20% of vintages, untouched until final evaluation.

If a reliable application date is unavailable, use stratified random splitting temporarily and state this limitation clearly in the portfolio.

### Success Criteria

- Test-set rows never affect imputation, binning, IV, encoding, tuning, or calibration.
- The same serialized pipeline can transform raw API input during inference.

## Phase 3 - Feature Engineering And Selection

### Goal

Improve predictive signal while preserving explainability and availability at decision time.

### Tasks

1. Create interpretable financial ratios, with denominator safeguards:
   - `loan_to_income = loan_amnt / annual_inc`.
   - `installment_to_income = installment / annual_inc`.
   - `revolving_balance_to_income = revol_bal / annual_inc`.
   - Credit-history aggregates where fields are reliably available.
2. Keep a missingness indicator for fields where missingness itself may contain signal.
3. Use IV and WoE primarily for the Logistic Regression scorecard baseline and feature diagnostics.
4. Do not automatically remove a tree-model feature solely because univariate IV is below `0.02`; compare model results with and without it.
5. Examine feature importance, SHAP summaries, and partial dependence for model reasonableness.
6. Check suspiciously high IV values and feature importance for leakage before accepting them.

### Success Criteria

- Feature engineering uses only approved pre-origination columns.
- Each engineered feature has a financial interpretation.
- Candidate feature sets are versioned and compared experimentally.

## Phase 4 - Model Benchmarking And Tuning

### Goal

Select a model based on robust holdout performance, calibration, explainability, and operational cost.

### Baselines

1. Logistic Regression with WoE-transformed variables.
2. LightGBM with expanded safe feature set.
3. XGBoost with expanded safe feature set.
4. CatBoost with native categorical features, if dependency and runtime are acceptable.

### Tuning

1. Use Optuna on the training period with stratified cross-validation or temporal folds.
2. Optimize ROC-AUC initially; record KS, Gini, PR-AUC, Brier score, training time, and model size for every trial.
3. Tune at least:
   - `n_estimators` and `learning_rate`.
   - Tree depth / leaves.
   - Row and feature subsampling.
   - Minimum child samples / regularization.
   - Class weighting.
4. Use early stopping with a validation fold where the selected library supports it.
5. Compare weighted and unweighted models. Class weighting may improve minority recall but can worsen probability calibration.

### Important Note On Deep Learning

Do not make deep learning the primary next experiment. A neural network is only a useful optional benchmark after the tabular baselines are strong. It is more appropriate when the project later includes unstructured documents, transaction sequences, text, or other high-dimensional data.

### Success Criteria

- A model comparison table exists for all baselines on the same out-of-time test set.
- The chosen model improves materially over the existing LightGBM baseline without leakage.

## Phase 5 - Probability Calibration

### Goal

Ensure model output can be interpreted as Probability of Default for expected-loss calculations.

### Tasks

1. Fit calibration only after model selection, using validation data or cross-validation within the training period.
2. Compare uncalibrated output, sigmoid calibration, and isotonic calibration.
3. Evaluate calibration with Brier score, calibration curve, and observed-versus-predicted default rates by decile.
4. Use the calibrated PD for Expected Loss, not raw classification output.

### Success Criteria

- Calibration improves or maintains Brier score on the untouched out-of-time test set.
- PD deciles have sensible monotonic observed default rates.

## Phase 6 - Business Decision Layer

### Goal

Convert predicted PD into an explainable approval policy rather than using an arbitrary `0.5` threshold.

### Tasks

1. Define assumptions for EAD, LGD, interest income, funding cost, operational cost, and expected profit.
2. Calculate per-application expected loss:

```text
Expected Loss = PD x LGD x EAD
```

3. Evaluate a grid of PD cutoffs and report:
   - Approval rate.
   - Default rate among approved loans.
   - Expected loss.
   - Expected profit or margin.
   - False-negative and false-positive counts.
4. Select a cutoff aligned with the chosen business objective and risk appetite.
5. Build a decile-based risk report that product or risk stakeholders can understand.

### Success Criteria

- Threshold selection is justified by financial outcomes, not `0.5` by default.
- The project produces a reproducible cutoff/profit matrix.

## Phase 7 - Portfolio-Ready Deliverables

### Code Deliverables

- A scikit-learn-compatible preprocessing and model pipeline.
- Reproducible configuration for split dates, feature set, model parameters, and random seed.
- Persisted experiment metrics in CSV or JSON.
- Unit tests for target mapping, leakage exclusions, feature transformations, and score scaling.
- A model artifact containing preprocessing, estimator, calibration, and input schema.

### Documentation Deliverables

- Feature inventory and leakage policy.
- Experiment comparison table.
- Final model card covering intended use, data, features, metrics, limitations, fairness considerations, and monitoring plan.
- Architecture diagram updated to match actual implementation.
- Clear statement that Lending Club data is a historical benchmark and not a production lending policy.

### Dashboard / API Deliverables

- FastAPI endpoint accepting only approved pre-origination fields.
- Input validation with Pydantic.
- Streamlit underwriting simulator showing PD, score, top SHAP drivers, approval decision, and expected loss.
- Cutoff optimizer page showing approval-rate and expected-profit trade-offs.

## Suggested Implementation Order

1. Audit data dictionary and resolve `fico_score` artifact mismatch.
2. Add approved pre-origination features to ETL and regenerate processed data.
3. Implement out-of-time split and leakage-safe preprocessing pipeline.
4. Build Logistic Regression + WoE and LightGBM baselines.
5. Add Optuna tuning and compare LightGBM, XGBoost, and CatBoost.
6. Calibrate the shortlisted model and evaluate it on the untouched test set.
7. Build the profit/cutoff analysis.
8. Add explainability, API, Streamlit dashboard, tests, and final documentation.

## Decision Gates

| Gate | Decision Rule |
|---|---|
| Data readiness | Proceed only after all model features have passed the leakage audit. |
| Feature readiness | Keep a feature set only if it improves validation metrics and remains explainable. |
| Model selection | Choose the simplest model with the strongest out-of-time discrimination and acceptable calibration. |
| Calibration | Retain a calibrator only if it improves Brier score and decile reliability on holdout data. |
| Business policy | Choose the threshold that meets the stated risk appetite and maximizes the agreed financial objective. |

## Final Definition Of Done

- Final out-of-time metrics are reported with no preprocessing leakage.
- The selected model is compared fairly with interpretable and boosting baselines.
- PD is calibrated and connected to Expected Loss.
- Approval cutoff is supported by a business trade-off analysis.
- Model, preprocessing, API schema, dashboard, and documentation use the same feature contract.
- Limitations and validation assumptions are explicit in the final portfolio case study.
