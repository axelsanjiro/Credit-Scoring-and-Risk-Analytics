import pandas as pd
import numpy as np
import os

def clean_lending_club_data(raw_data_path: str, processed_data_path: str):
    print(f"Loading raw data from {raw_data_path}...")
    
    try:
        # Kita gunakan 'usecols' agar RAM tidak jebol karena membaca 150+ kolom yang tidak dipakai
        columns_to_keep = [
            'loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 
            'revol_util', 'delinq_2yrs', 'inq_last_6mths', 'emp_length', 
            'home_ownership', 'purpose', 'loan_status'
        ]
        df = pd.read_csv(raw_data_path, usecols=lambda c: c in columns_to_keep, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {raw_data_path}.")
        print("Silakan unduh 'loan.csv' dari Kaggle Lending Club dan letakkan di data/raw/")
        return

    print("Data berhasil dimuat. Memulai proses Data Cleaning & Wrangling...")
    
    # 1. Filter Target Variable (Hanya ambil pinjaman yang siklusnya sudah selesai)
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    
    # Mapping Target: 1 = Gagal Bayar (Bad Loan), 0 = Lancar (Good Loan)
    df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
    df = df.drop(columns=['loan_status'])
    
    # 2. Penanganan Missing Values (Imputasi sederhana)
    df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())
    df['dti'] = df['dti'].fillna(df['dti'].median())
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
    df['delinq_2yrs'] = df['delinq_2yrs'].fillna(0)
    df['inq_last_6mths'] = df['inq_last_6mths'].fillna(0)
    
    # 3. Ekstraksi String ke Numerik (contoh: "10+ years" menjadi 10.0)
    df['emp_length_years'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    df['emp_length_years'] = df['emp_length_years'].fillna(0) # Asumsi 0 jika kosong
    df = df.drop(columns=['emp_length'])
    
    # 4. Standardisasi Penamaan Kolom (Menyesuaikan dengan arsitektur awal kita)
    df = df.rename(columns={'fico_range_low': 'fico_score', 'purpose': 'loan_purpose'})
    
    # 5. Buang sisa baris yang masih mengandung NaN (jika ada) untuk keamanan model
    df = df.dropna()
    
    # 6. Simpan Data Bersih
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    
    print(f"Data cleaning selesai! Dataset diproses dan disimpan ke {processed_data_path}")
    print(f"Total baris valid: {len(df):,}. Default rate (Gagal Bayar): {df['default'].mean():.2%}")

if __name__ == "__main__":
    RAW_PATH = "data/raw/loan.csv"
    PROCESSED_PATH = "data/processed/credit_applications_cleaned.csv"
    clean_lending_club_data(RAW_PATH, PROCESSED_PATH)