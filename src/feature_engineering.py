import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    Menghitung Weight of Evidence (WoE) dan Information Value (IV).
    Membagi data kontinu ke dalam kelompok (bins) untuk mengevaluasi risiko per kelompok.
    """
    data = df[[feature, target]].copy()
    
    # Binning (Pengelompokan)
    if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > bins:
        data['bin'] = pd.qcut(data[feature], q=bins, duplicates='drop')
    else:
        data['bin'] = data[feature].astype(str)
        
    grouped = data.groupby('bin', observed=False)[target].agg(['count', 'sum']).reset_index()
    grouped.columns = ['bin', 'total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    grouped['dist_good'] = grouped['good'] / total_good
    grouped['dist_bad'] = grouped['bad'] / total_bad
    
    # Mencegah error pembagian dengan nol (zero division)
    grouped['dist_good'] = np.where(grouped['dist_good'] == 0, 0.0001, grouped['dist_good'])
    grouped['dist_bad'] = np.where(grouped['dist_bad'] == 0, 0.0001, grouped['dist_bad'])
    
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    return grouped, total_iv

def prob_to_credit_score(prob_default: np.ndarray, base_score: int = 600, pdo: int = 20, base_odds: float = 50.0) -> np.ndarray:
    """
    Mengonversi Probabilitas Gagal Bayar (PD) menjadi standar Credit Score (FICO-like).
    Rumus: Score = Offset + Factor * ln(Odds)
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    # Membatasi probabilitas agar tidak terjadi log(0)
    pd_clipped = np.clip(prob_default, 0.0001, 0.9999)
    odds = (1.0 - pd_clipped) / pd_clipped
    scores = offset + factor * np.log(odds)
    
    # Batas standar industri: minimum 300, maksimum 850
    return np.clip(np.round(scores), 300, 850).astype(int)

def select_features_by_iv(df: pd.DataFrame, target: str, threshold: float = 0.02) -> List[str]:
    """
    Mengevaluasi semua fitur dan membuang (drop) fitur yang Information Value (IV)-nya 
    berada di bawah threshold standar industri.
    """
    features = [col for col in df.columns if col != target]
    selected_features = []
    
    print(f"--- EVALUASI INFORMATION VALUE (IV) ---")
    for feat in features:
        _, iv = calculate_woe_iv(df, feat, target)
        
        # Rule of Thumb Industri Perbankan
        if iv < 0.02:
            status = "DITOLAK (Useless)"
        elif iv < 0.1:
            status = "DITERIMA (Weak)"
        elif iv < 0.3:
            status = "DITERIMA (Medium)"
        elif iv < 0.5:
            status = "DITERIMA (Strong)"
        else:
            status = "DITERIMA (Suspicious/Too Good)"
            
        print(f"Fitur: {feat.ljust(20)} | IV: {iv:.4f} | {status}")
        
        if iv >= threshold:
            selected_features.append(feat)
            
    return selected_features

if __name__ == "__main__":
    PROCESSED_PATH = "data/processed/credit_applications_cleaned.csv"
    SELECTED_PATH = "data/processed/credit_applications_selected.csv"
    
    print("Memuat data bersih...")
    try:
        df = pd.read_csv(PROCESSED_PATH)
    except FileNotFoundError:
        print(f"File {PROCESSED_PATH} tidak ditemukan. Jalankan src/data_pipeline.py dulu.")
        exit()
        
    target_col = 'default'
    
    # 1. Jalankan Feature Selection
    selected_cols = select_features_by_iv(df, target=target_col, threshold=0.02)
    
    # 2. Filter dataset hanya dengan fitur yang lolos seleksi
    final_cols = selected_cols + [target_col]
    df_final = df[final_cols]
    
    # 3. Simpan data yang siap dimasukkan ke Model ML
    os.makedirs(os.path.dirname(SELECTED_PATH), exist_ok=True)
    df_final.to_csv(SELECTED_PATH, index=False)
    
    print(f"\n--- HASIL FEATURE SELECTION ---")
    print(f"Berhasil menyeleksi {len(selected_cols)} fitur tangguh dari {len(df.columns)-1} fitur awal.")
    print(f"Data final untuk tahapan Modeling disimpan di: {SELECTED_PATH}")