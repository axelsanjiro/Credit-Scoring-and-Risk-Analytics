import json
import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "default"
DATE_COLUMN = "issue_date"
# Pricing outputs require a separate, decision-point-specific experiment.
CONDITIONAL_FEATURES = {"int_rate", "installment"}
FEATURE_SET_VERSION = "allowed_preorigination_v1"


@dataclass
class TemporalSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    summary: dict


def load_temporal_splits(
    data_path: str,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
) -> TemporalSplits:
    """Create chronological train, validation, and untouched test partitions."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test period.")

    df = pd.read_csv(data_path, parse_dates=[DATE_COLUMN])
    if TARGET_COLUMN not in df or DATE_COLUMN not in df:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' and '{DATE_COLUMN}'.")
    if df[DATE_COLUMN].isna().any():
        raise ValueError(f"'{DATE_COLUMN}' contains missing or invalid dates.")

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    vintages = pd.Index(df[DATE_COLUMN].unique())
    train_end = int(len(vintages) * train_fraction)
    validation_end = int(len(vintages) * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end == train_end or validation_end == len(vintages):
        raise ValueError("Insufficient application vintages for requested temporal split.")

    train_dates = vintages[:train_end]
    validation_dates = vintages[train_end:validation_end]
    test_dates = vintages[validation_end:]
    feature_columns = [
        column for column in df.columns
        if column not in {TARGET_COLUMN, DATE_COLUMN, *CONDITIONAL_FEATURES}
    ]

    def partition(dates: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
        rows = df[DATE_COLUMN].isin(dates)
        features = add_financial_features(df.loc[rows, feature_columns])
        return features, df.loc[rows, TARGET_COLUMN]

    X_train, y_train = partition(train_dates)
    X_validation, y_validation = partition(validation_dates)
    X_test, y_test = partition(test_dates)
    unobserved_train_features = X_train.columns[X_train.isna().all()].tolist()
    if unobserved_train_features:
        X_train = X_train.drop(columns=unobserved_train_features)
        X_validation = X_validation.drop(columns=unobserved_train_features)
        X_test = X_test.drop(columns=unobserved_train_features)
    summary = {
        "feature_count": len(X_train.columns),
        "feature_set_version": FEATURE_SET_VERSION,
        "excluded_conditional_features": sorted(CONDITIONAL_FEATURES),
        "dropped_no_train_observations": unobserved_train_features,
        "train": split_summary(train_dates, y_train),
        "validation": split_summary(validation_dates, y_validation),
        "test": split_summary(test_dates, y_test),
    }
    return TemporalSplits(X_train, y_train, X_validation, y_validation, X_test, y_test, summary)


def add_financial_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add approved, deterministic affordability and credit-capacity ratios."""
    features = features.copy()
    annual_income = features["annual_inc"].where(features["annual_inc"] > 0)
    features["loan_to_income"] = features["loan_amnt"] / annual_income
    features["revolving_balance_to_income"] = features["revol_bal"] / annual_income

    revolving_limit = features["total_rev_hi_lim"].where(features["total_rev_hi_lim"] > 0)
    features["revolving_balance_to_limit"] = features["revol_bal"] / revolving_limit

    credit_history_years = features["credit_history_months"].where(
        features["credit_history_months"] > 0
    ) / 12
    features["accounts_per_credit_history_year"] = features["total_acc"] / credit_history_years
    return features


def split_summary(dates: pd.Index, target: pd.Series) -> dict:
    return {
        "rows": int(len(target)),
        "vintages": int(len(dates)),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "default_rate": float(target.mean()),
    }


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if len(numeric_features) + len(categorical_features) != len(X_train.columns):
        raise ValueError("All model features must be numeric or categorical.")

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def fit_and_save_preprocessor(data_path: str, artifact_path: str, summary_path: str) -> TemporalSplits:
    splits = load_temporal_splits(data_path)
    preprocessor = build_preprocessor(splits.X_train)
    preprocessor.fit(splits.X_train)

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(preprocessor, artifact_path)
    with open(summary_path, "w", encoding="ascii") as file:
        json.dump(splits.summary, file, indent=2)

    return splits


if __name__ == "__main__":
    splits = fit_and_save_preprocessor(
        "data/processed/credit_applications_cleaned.csv",
        "models/credit_preprocessor.pkl",
        "models/temporal_split_summary.json",
    )
    for split_name in ("train", "validation", "test"):
        details = splits.summary[split_name]
        print(
            f"{split_name.title()}: {details['rows']:,} rows, "
            f"{details['start']} to {details['end']}, "
            f"default rate {details['default_rate']:.2%}"
        )
