import json
import os
import time
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

from src.modeling_pipeline import build_preprocessor, load_temporal_splits


RANDOM_STATE = 42
DATA_PATH = "data/processed/credit_applications_cleaned.csv"
ARTIFACT_PATH = "models/credit_risk_model.pkl"
METRICS_PATH = "models/experiment_metrics.csv"
CUTOFF_PATH = "models/cutoff_analysis.csv"
DECILE_PATH = "models/risk_deciles.csv"
IMPORTANCE_PATH = "models/feature_importance.csv"
SHAP_PATH = "models/shap_summary.csv"
MAX_APPROVED_DEFAULT_RATE = 0.15


@dataclass
class CalibratedCreditModel:
    """Predict calibrated PD from raw approved feature columns."""

    pipeline: Pipeline
    calibrator: object
    calibration_method: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = self.pipeline.predict_proba(X)[:, 1]
        if self.calibration_method == "sigmoid":
            probability = self.calibrator.predict_proba(probability.reshape(-1, 1))[:, 1]
        else:
            probability = self.calibrator.predict(probability)
        return np.column_stack((1 - probability, probability))


# Keep persisted artifacts importable when this module runs with ``python -m``.
CalibratedCreditModel.__module__ = "src.experiment"


def calculate_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    good_probability = probability[y_true.to_numpy() == 0]
    bad_probability = probability[y_true.to_numpy() == 1]
    auc = roc_auc_score(y_true, probability)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(ks_2samp(good_probability, bad_probability).statistic),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }


def create_estimator(name: str, params: dict | None = None):
    params = params or {}
    if name == "logistic_regression":
        return LogisticRegression(max_iter=300, solver="lbfgs", random_state=RANDOM_STATE)
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 250),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            min_child_samples=params.get("min_child_samples", 100),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        )
    raise ValueError(f"Unsupported model: {name}.")


def make_pipeline(name: str, X_train: pd.DataFrame, params: dict | None = None) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", create_estimator(name, params)),
    ])


def benchmark_models(splits) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    records = []
    fitted_models = {}
    for name in ("logistic_regression", "lightgbm", "xgboost", "catboost"):
        pipeline = make_pipeline(name, splits.X_train)
        started = time.perf_counter()
        pipeline.fit(splits.X_train, splits.y_train)
        metrics = calculate_metrics(splits.y_validation, pipeline.predict_proba(splits.X_validation)[:, 1])
        records.append({"model": name, "stage": "baseline", "seconds": time.perf_counter() - started, **metrics})
        fitted_models[name] = pipeline
    return pd.DataFrame(records), fitted_models


def tune_lightgbm(splits, trials: int = 4) -> tuple[dict, pd.DataFrame]:
    """Tune LightGBM with expanding temporal folds from training vintages only."""
    folds = TimeSeriesSplit(n_splits=3)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        scores = []
        for train_index, fold_index in folds.split(splits.X_train):
            X_fold_train = splits.X_train.iloc[train_index]
            model = make_pipeline("lightgbm", X_fold_train, params)
            model.fit(X_fold_train, splits.y_train.iloc[train_index])
            probability = model.predict_proba(splits.X_train.iloc[fold_index])[:, 1]
            scores.append(roc_auc_score(splits.y_train.iloc[fold_index], probability))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    trials_frame = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return study.best_params, trials_frame


def fit_calibrator(model: Pipeline, X_validation: pd.DataFrame, y_validation: pd.Series):
    """Select calibration on a later validation period, then fit it on all validation rows."""
    split_index = int(len(X_validation) * 2 / 3)
    if split_index == 0 or split_index == len(X_validation):
        raise ValueError("Validation data must contain rows for calibration selection.")

    validation_probability = model.predict_proba(X_validation)[:, 1]
    selection_probability = validation_probability[:split_index]
    selection_target = y_validation.iloc[:split_index]
    evaluation_probability = validation_probability[split_index:]
    evaluation_target = y_validation.iloc[split_index:]
    calibrators = {
        "sigmoid": LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE).fit(
            selection_probability.reshape(-1, 1), selection_target
        ),
        "isotonic": IsotonicRegression(out_of_bounds="clip").fit(
            selection_probability, selection_target
        ),
    }
    scores = {}
    for name, calibrator in calibrators.items():
        if name == "sigmoid":
            probability = calibrator.predict_proba(evaluation_probability.reshape(-1, 1))[:, 1]
        else:
            probability = calibrator.predict(evaluation_probability)
        scores[name] = brier_score_loss(evaluation_target, probability)

    method = min(scores, key=scores.get)
    if method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE).fit(
            validation_probability.reshape(-1, 1), y_validation
        )
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(validation_probability, y_validation)
    return CalibratedCreditModel(model, calibrator, method), scores


def cutoff_analysis(X_test: pd.DataFrame, y_test: pd.Series, probability: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate PD cutoffs under documented, portfolio-level assumptions."""
    ead = X_test["loan_amnt"].to_numpy()
    term_years = X_test["term_months"].to_numpy() / 12
    lgd = 0.45
    annual_margin = 0.07
    operational_cost = 100.0
    records = []
    for cutoff in np.arange(0.05, 0.76, 0.01):
        approved = probability < cutoff
        approved_count = int(approved.sum())
        expected_loss = probability[approved] * lgd * ead[approved]
        expected_income = annual_margin * ead[approved] * term_years[approved]
        expected_profit = expected_income - expected_loss - operational_cost
        records.append({
            "pd_cutoff": round(float(cutoff), 2),
            "approval_rate": float(approved.mean()),
            "approved_default_rate": float(y_test.to_numpy()[approved].mean()) if approved_count else np.nan,
            "expected_loss": float(expected_loss.sum()),
            "expected_profit": float(expected_profit.sum()),
            "false_negatives": int(((y_test.to_numpy() == 1) & approved).sum()),
            "false_positives": int(((y_test.to_numpy() == 0) & ~approved).sum()),
        })

    deciles = pd.DataFrame({"default": y_test.to_numpy(), "pd": probability})
    deciles["decile"] = pd.qcut(deciles["pd"].rank(method="first"), 10, labels=False) + 1
    risk_report = deciles.groupby("decile", observed=True).agg(
        applications=("default", "size"), predicted_pd=("pd", "mean"), observed_default_rate=("default", "mean")
    ).reset_index()
    return pd.DataFrame(records), risk_report


def save_feature_importance(model: Pipeline) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": estimator.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(IMPORTANCE_PATH, index=False)


def save_shap_summary(model: Pipeline, X_validation: pd.DataFrame, sample_size: int = 1_000) -> None:
    """Save mean absolute SHAP values for tree-model reasonableness review."""
    if not hasattr(model.named_steps["model"], "feature_importances_"):
        return
    import shap

    sample = X_validation.iloc[:sample_size]
    transformed = model.named_steps["preprocessor"].transform(sample)
    values = shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed)
    summary = pd.DataFrame({
        "feature": model.named_steps["preprocessor"].get_feature_names_out(),
        "mean_absolute_shap": np.abs(values).mean(axis=0),
    })
    summary.sort_values("mean_absolute_shap", ascending=False).to_csv(SHAP_PATH, index=False)


def run_experiments() -> dict:
    os.makedirs("models", exist_ok=True)
    splits = load_temporal_splits(DATA_PATH)
    metrics, models = benchmark_models(splits)
    best_params, trials = tune_lightgbm(splits)
    tuned_lightgbm = make_pipeline("lightgbm", splits.X_train, best_params)
    started = time.perf_counter()
    tuned_lightgbm.fit(splits.X_train, splits.y_train)
    tuned_metrics = calculate_metrics(
        splits.y_validation, tuned_lightgbm.predict_proba(splits.X_validation)[:, 1]
    )
    metrics = pd.concat([metrics, pd.DataFrame([{
        "model": "lightgbm", "stage": "optuna_tuned", "seconds": time.perf_counter() - started, **tuned_metrics
    }])], ignore_index=True)
    models["lightgbm_optuna_tuned"] = tuned_lightgbm
    metrics.to_csv(METRICS_PATH, index=False)
    trials.to_csv("models/optuna_trials.csv", index=False)

    winner = metrics.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    winner_name = winner["model"] if winner["stage"] == "baseline" else "lightgbm_optuna_tuned"
    model = models[winner_name]
    calibrated_model, calibration_scores = fit_calibrator(model, splits.X_validation, splits.y_validation)
    test_probability = calibrated_model.predict_proba(splits.X_test)[:, 1]
    test_metrics = calculate_metrics(splits.y_test, test_probability)
    cutoffs, deciles = cutoff_analysis(splits.X_test, splits.y_test, test_probability)
    cutoffs.to_csv(CUTOFF_PATH, index=False)
    deciles.to_csv(DECILE_PATH, index=False)
    save_feature_importance(model)
    save_shap_summary(model, splits.X_validation)
    joblib.dump(calibrated_model, ARTIFACT_PATH)

    acceptable_cutoffs = cutoffs[cutoffs["approved_default_rate"] <= MAX_APPROVED_DEFAULT_RATE]
    best_cutoff = acceptable_cutoffs.loc[acceptable_cutoffs["expected_profit"].idxmax()].to_dict()
    report = {
        "selected_model": winner_name,
        "validation_metrics": winner.drop(labels=["model", "stage"]).to_dict(),
        "calibration_selection_brier_by_method": calibration_scores,
        "selected_calibration": calibrated_model.calibration_method,
        "test_metrics": test_metrics,
        "recommended_cutoff": best_cutoff,
        "split_summary": splits.summary,
        "assumptions": {
            "lgd": 0.45,
            "annual_margin": 0.07,
            "operational_cost": 100.0,
            "max_approved_default_rate": MAX_APPROVED_DEFAULT_RATE,
        },
    }
    with open("models/final_model_report.json", "w", encoding="ascii") as file:
        json.dump(report, file, indent=2)
    return report


if __name__ == "__main__":
    from src.experiment import run_experiments as imported_run_experiments

    report = imported_run_experiments()
    print(f"Selected model: {report['selected_model']}")
    print(f"Calibration: {report['selected_calibration']}")
    print(f"Test ROC-AUC: {report['test_metrics']['roc_auc']:.4f}")
    print(f"Recommended PD cutoff: {report['recommended_cutoff']['pd_cutoff']:.2f}")
