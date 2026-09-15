# Feature Availability And Leakage Inventory

## Purpose

This inventory defines which Lending Club fields may enter the credit-risk modelling pipeline. The decision criterion is feature availability at the intended underwriting decision point, not predictive strength alone.

Definitions:

- **Allowed:** available at application or origination and eligible for modelling.
- **Conditional:** potentially usable, but requires a decision-point or data-quality check before modelling.
- **Excluded:** identifier, target, post-origination outcome, servicing, recovery, hardship, or settlement field.
- **Unavailable:** not present in the supplied raw dataset.

## Current Pipeline Features

| Feature | Data dictionary meaning | Availability | Decision | Rationale |
|---|---|---|---|---|
| `loan_amnt` | Requested loan amount | Application | Allowed | Applicant request known before approval. |
| `int_rate` | Interest rate on the loan | Underwriting / pricing | Conditional | Usable only after pricing is assigned. Do not use when the model itself sets pricing. |
| `annual_inc` | Self-reported annual income | Registration | Allowed | Available in the application. |
| `dti` | Debt-to-income ratio | Application / bureau | Allowed | Available before loan decision. |
| `revol_util` | Revolving credit utilization | Credit bureau | Allowed | Credit-file attribute at decision time. |
| `delinq_2yrs` | Delinquencies in prior two years | Credit bureau | Allowed | Historical borrower attribute. |
| `inq_last_6mths` | Credit inquiries in prior six months | Credit bureau | Allowed | Historical borrower attribute. |
| `emp_length` | Employment duration | Registration | Allowed | Application-time attribute. |
| `home_ownership` | Borrower housing status | Registration / bureau | Allowed | Available before approval. |
| `purpose` | Borrower-stated purpose | Application | Allowed | Available before approval. |
| `loan_status` | Current status of the loan | Outcome | Excluded from predictors | Used only to construct the target. |
| `issue_d` | Month in which loan was funded | Origination | Conditional | Retain only for chronological splitting and monitoring, not as a predictor by default. |

## Approved Candidate Features For The Next ETL Version

These fields are in `data/raw/loan.csv`, have an application-time or credit-bureau interpretation in the data dictionary, and should be evaluated in the next pipeline iteration.

| Feature group | Features | Availability | Decision | Rationale |
|---|---|---|---|---|
| Loan terms | `term`, `installment` | Application / origination | Allowed | Requested term and resulting payment obligation support affordability analysis. |
| Lending Club assessment | `grade`, `sub_grade` | Underwriting | Conditional | Strong risk signal, but may duplicate Lending Club's existing model. Use only when mimicking a post-underwriting decision. |
| Income verification | `verification_status` | Underwriting | Allowed | Available during verification. |
| Credit history | `earliest_cr_line`, `open_acc`, `total_acc`, `mort_acc`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens` | Credit bureau | Allowed | Historical credit depth and adverse-record signals. |
| Revolving credit | `revol_bal`, `total_rev_hi_lim`, `bc_open_to_buy`, `bc_util`, `percent_bc_gt_75` | Credit bureau | Allowed | Utilization, capacity, and balance signals. |
| Recent activity | `acc_open_past_24mths`, `open_rv_12m`, `open_rv_24m`, `num_tl_op_past_12m`, `mths_since_recent_inq` | Credit bureau | Allowed | Recent borrowing and inquiry behavior. |
| Delinquency history | `chargeoff_within_12_mths`, `num_accts_ever_120_pd`, `num_tl_90g_dpd_24m`, `pct_tl_nvr_dlq` | Credit bureau | Allowed | Historical adverse-credit behavior. |
| Account composition | `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats` | Credit bureau | Allowed | Composition and utilization of the borrower's credit portfolio. |
| Aggregate balances | `avg_cur_bal`, `tot_cur_bal`, `total_bal_ex_mort`, `tot_hi_cred_lim`, `total_bc_limit`, `total_il_high_credit_limit` | Credit bureau | Allowed | Credit capacity and aggregate balance information. |
| Application type | `application_type`, `initial_list_status` | Origination | Conditional | Available at origination, but business meaning and missingness must be reviewed. |

Before use, each candidate must pass missingness, distribution, and out-of-time stability checks. Categorical variables require explicit unknown-category handling. Numeric variables require plausibility checks and training-only imputation.

## Conditional Or Excluded Candidate Features

| Feature group | Features | Decision | Rationale |
|---|---|---|---|
| Lending Club pricing outputs | `int_rate`, `grade`, `sub_grade`, `installment` | Conditional | May reflect the platform's own underwriting and pricing policy. Valid only if available before the selected decision point. |
| Geographic data | `addr_state`, `zip_code` | Excluded for initial model | High-cardinality / geographic proxy risk. Assess only later for descriptive monitoring and fairness review. |
| Free text | `emp_title`, `title`, `desc` | Excluded for initial model | High-cardinality and unstructured input; requires separate NLP design and validation. |
| Joint-application data | `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`, `sec_app_*` | Excluded for initial model | Structurally missing for individual loans. Include only with an explicit joint-application feature design. |
| Delinquency snapshot fields | `num_tl_30dpd`, `acc_now_delinq`, `delinq_amnt` | Conditional | Verify exact as-of timing and business suitability before use. |
| Collection aggregate | `tot_coll_amt` | Conditional | Dictionary describes total collection amounts ever owed; retain only if confirmed as historical bureau data at application time. |
| Data-quality flags | `policy_code`, `pymnt_plan`, `url`, `id`, `member_id` | Excluded | Constant, operational, URL, or identifier fields provide no valid generalizable underwriting signal. |

## Explicit Leakage Exclusions

The following groups must never enter a borrower-default prediction model because they become known after the loan is issued or are derived from the outcome.

| Leakage group | Examples | Reason |
|---|---|---|
| Outstanding and funded amounts | `funded_amnt`, `funded_amnt_inv`, `out_prncp`, `out_prncp_inv` | Funding and remaining-principal information is not consistently known at the application decision point. |
| Repayment outcomes | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Directly reflect repayment after origination. |
| Recovery and collection | `recoveries`, `collection_recovery_fee` | Result from default or post-origination servicing. |
| Loan monitoring | `last_credit_pull_d` | Can reveal post-origination account observation activity. |
| Hardship workflow | `hardship_*`, `deferral_term`, `payment_plan_start_date`, `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status` | Captured after borrower repayment difficulty emerges. |
| Debt settlement workflow | `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_*` | Captured after delinquency/default and collection activity. |
| Target and outcome fields | `loan_status` | Used only for final target construction. |

## Unavailable Fields

The supplied `data/raw/loan.csv` does not contain `fico_range_low` or `fico_range_high`. Therefore `fico_score` cannot currently be created from this source. The previous ETL whitelist included `fico_range_low`, but `pandas.read_csv(..., usecols=lambda ...)` silently skipped the absent column.

This was a documentation and data-contract mismatch, not an imputation issue. It has been removed from the ETL whitelist and current documentation. The next ETL version will use the approved alternative credit-profile fields in this inventory.

## Audit Decisions Before ETL Changes

1. Use the Allowed features as the candidate set for the next ETL version.
2. Preserve `issue_d` only for a chronological split.
3. Keep Conditional features in a separate experiment group with explicit decision-point justification.
4. Exclude every leakage group regardless of univariate predictive strength.
5. Do not use geographic, text, or joint-application fields in the first improved baseline.
