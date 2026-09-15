import pandas as pd

from src.modeling_pipeline import add_financial_features, load_temporal_splits
from src.scoring import prob_to_credit_score


def test_temporal_splits_do_not_overlap():
    splits = load_temporal_splits("data/processed/credit_applications_cleaned.csv")
    summary = splits.summary
    assert summary["train"]["end"] < summary["validation"]["start"]
    assert summary["validation"]["end"] < summary["test"]["start"]
    assert "int_rate" not in splits.X_train
    assert "issue_date" not in splits.X_train


def test_financial_ratios_handle_zero_income():
    features = pd.DataFrame({
        "loan_amnt": [10_000], "annual_inc": [0], "revol_bal": [500],
        "total_rev_hi_lim": [1_000], "credit_history_months": [120], "total_acc": [10],
    })
    engineered = add_financial_features(features)
    assert pd.isna(engineered.loc[0, "loan_to_income"])
    assert engineered.loc[0, "revolving_balance_to_limit"] == 0.5


def test_credit_score_range_and_order():
    scores = prob_to_credit_score(pd.Series([0.01, 0.50]).to_numpy())
    assert 300 <= scores[1] < scores[0] <= 850
