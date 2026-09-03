import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Menghitung Kolmogorov-Smirnov (KS) Statistic.
    Mengukur jarak pemisah maksimum antara distribusi kumulatif Good vs Bad.
    """
    good_probs = y_prob[y_true == 0]
    bad_probs = y_prob[y_true == 1]
    ks_stat, _ = ks_2samp(good_probs, bad_probs)
    return float(ks_stat)

def train_and_evaluate():
    input_path = "data/processed/credit_applications_selected.csv"
    if not os.path.exists(input_path):
        # Fallback jika baru menjalankan sampai tahap cleaned
        input_path = "data/processed/credit_applications_cleaned.csv"
        
    print(f"Memuat dataset pelatihan dari: {input_path}")
    df = pd.read_csv(input_path)
    
    target_col = "default"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan.")

    # 1. ENCODING CATEGORICAL VARIABLES
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        print(f"Melakukan One-Hot Encoding pada fitur kategorikal: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # 2. STRATIFIED TRAIN-TEST SPLIT
    # Memastikan rasio default tetap seimbang di train dan test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data latih: {len(X_train):,} baris | Data uji: {len(X_test):,} baris")

    # 3. PENANGANAN CLASS IMBALANCE
    # Memberi penalti lebih berat pada kesalahan klasifikasi nasabah default
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos = negative_count / positive_count
    print(f"Class ratio: {scale_pos:.2f}:1 (Mengonfigurasi scale_pos_weight = {scale_pos:.2f})")

    # 4. MODEL TRAINING DENGAN LIGHTGBM
    base_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=-1
    )
    
    print("Melatih algoritma LightGBM...")
    base_model.fit(X_train, y_train)

    # 5. PROBABILITY CALIBRATION (ISOTONIC REGRESSION)
    # Menyelaraskan probabilitas output model agar sesuai dengan frekuensi empiris default
    print("Mengalibrasi probabilitas menggunakan Isotonic Regression")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, 
        method="isotonic", 
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    # 6. EVALUASI METRIK KHUSUS CREDIT RISK
    y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    gini = 2 * auc - 1
    ks = calculate_ks_statistic(y_test.values, y_pred_prob)
    brier = brier_score_loss(y_test, y_pred_prob)

    print("\n==============================================")
    print("      METRIK EVALUASI MODEL RISIKO KREDIT     ")
    print("==============================================")
    print(f"ROC-AUC Score       : {auc:.4f}  (Biro standard > 0.75)")
    print(f"Gini Coefficient    : {gini:.4f}  (Standar Basel II/III > 0.50)")
    print(f"KS-Statistic        : {ks*100:.2f}% (Daya pemisah target > 40%)")
    print(f"Brier Score (Loss)  : {brier:.4f}  (Mendekati 0.0 = Probabilitas terkalibrasi)")
    print("----------------------------------------------")
    print("\nClassification Report (Threshold 0.50):")
    print(classification_report(y_test, y_pred))

    # 7. EXPORT MODEL ARTIFACTS
    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated_model, "models/credit_lgbm_model.pkl")
    joblib.dump(feature_names, "models/model_features.pkl")
    print("Model terkalibrasi dan daftar fitur berhasil disimpan ke folder 'models/'.")

if __name__ == "__main__":
    train_and_evaluate()