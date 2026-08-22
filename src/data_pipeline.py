import pandas as pd
import numpy as np
import os

def clean_lending_club_data(raw_data_path: str, processed_data_path: str):
    print(f"Loading raw data from {raw_data_path}...")
    
    try:
        # 1. Memory Optimization: Hanya memuat 12 kolom Pre-Origination yang aman dari Data Leakage
        # dan memiliki Missing Value di bawah 50% berdasarkan hasil EDA.
        columns_to_keep = [
            'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 
            'revol_util', 'delinq_2yrs', 'inq_last_6mths', 'emp_length', 
            'home_ownership', 'purpose', 'loan_status'
        ]
        df = pd.read_csv(raw_data_path, usecols=lambda c: c in columns_to_keep, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {raw_data_path}.")
        return

    print("Data berhasil dimuat. Memulai proses Data Cleaning & Wrangling...")
    
    # 2. TARGET DEFINITION: Buang pinjaman 'Current' (masih berjalan)
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    
    # Mapping Target: 1 = Gagal Bayar (Bad Loan), 0 = Lancar (Good Loan)
    df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
    df = df.drop(columns=['loan_status'])
    
    # 3. TEXT TO NUMERIC (REGEX & STRING PARSING)
    # A. Membersihkan 'int_rate' (Suku Bunga) dari format ' 10.65%' menjadi float 10.65
    if df['int_rate'].dtype == 'O':
        df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').str.strip().astype(float)
        
    # B. Ekstraksi angka dari 'emp_length' (contoh: "10+ years" menjadi 10.0)
    df['emp_length_years'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
    df['emp_length_years'] = df['emp_length_years'].fillna(0) # Asumsi 0 jika kosong
    df = df.drop(columns=['emp_length'])
    
    # 4. MISSING VALUE IMPUTATION
    # Menggunakan median agar tahan terhadap outlier (seperti income yang sangat tinggi)
    df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())
    df['dti'] = df['dti'].fillna(df['dti'].median())
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
    df['int_rate'] = df['int_rate'].fillna(df['int_rate'].median())
    
    # Untuk fitur diskrit/hitungan, isi dengan 0
    df['delinq_2yrs'] = df['delinq_2yrs'].fillna(0)
    df['inq_last_6mths'] = df['inq_last_6mths'].fillna(0)
    
    # 5. STANDARDIZASI NAMA KOLOM
    df = df.rename(columns={'fico_range_low': 'fico_score', 'purpose': 'loan_purpose'})
    
    # 6. SAFETY DROP: Buang sisa baris yang masih mengandung NaN di kolom kategorikal
    df = df.dropna()
    
    # 7. EXPORT DATA BERSIH
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    
    print(f"\n--- PROSES SELESAI ---")
    print(f"Dataset diproses dan disimpan ke: {processed_data_path}")
    print(f"Total baris valid: {len(df):,}")
    print(f"Default rate (Gagal Bayar): {df['default'].mean():.2%}")

if __name__ == "__main__":
    RAW_PATH = "data/raw/loan.csv"
    PROCESSED_PATH = "data/processed/credit_applications_cleaned.csv"
    clean_lending_club_data(RAW_PATH, PROCESSED_PATH)
