import pandas as pd
import numpy as np
import os

def clean_lending_club_data(raw_data_path: str, processed_data_path: str):
    print(f"Loading raw data from {raw_data_path}...")
    
    try:
        # 1. Load audited application-time and credit-bureau fields only.
        # issue_d is retained for the future chronological split, not as a model predictor.
        columns_to_keep = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
            'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
            'loan_status', 'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
            'total_acc', 'tot_cur_bal', 'open_rv_12m', 'open_rv_24m',
            'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
            'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'mort_acc',
            'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
            'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
            'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
            'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
            'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens',
            'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
            'total_il_high_credit_limit'
        ]
        df = pd.read_csv(raw_data_path, usecols=columns_to_keep, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found: {raw_data_path}.")
        return

    print("Data loaded. Starting cleaning and wrangling...")
    
    # 2. TARGET DEFINITION: Exclude active loans with unknown final outcomes.
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    
    # Target mapping: 1 = bad loan, 0 = good loan.
    df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
    df = df.drop(columns=['loan_status'])
    
    # 3. Normalize application fields without learning statistics from the full dataset.
    df['int_rate'] = pd.to_numeric(
        df['int_rate'].astype(str).str.replace('%', '', regex=False).str.strip(),
        errors='coerce'
    )
    df['term_months'] = pd.to_numeric(
        df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['emp_length_years'] = pd.to_numeric(
        df['emp_length'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )

    issue_date = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    earliest_credit_date = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['issue_date'] = issue_date
    df['credit_history_months'] = (
        (issue_date.dt.year - earliest_credit_date.dt.year) * 12
        + issue_date.dt.month - earliest_credit_date.dt.month
    )

    # Missing numeric values are retained for training-only imputation in the next stage.
    df = df.drop(columns=['term', 'emp_length', 'issue_d', 'earliest_cr_line'])
    df = df.rename(columns={'purpose': 'loan_purpose'})
    
    # 4. EXPORT CLEANED, BUT NOT YET IMPUTED, DATA
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    
    print("\n--- PROCESS COMPLETE ---")
    print(f"Processed dataset saved to: {processed_data_path}")
    print(f"Total valid rows: {len(df):,}")
    print(f"Default rate: {df['default'].mean():.2%}")

if __name__ == "__main__":
    RAW_PATH = "data/raw/loan.csv"
    PROCESSED_PATH = "data/processed/credit_applications_cleaned.csv"
    clean_lending_club_data(RAW_PATH, PROCESSED_PATH)
