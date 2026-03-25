"""
train_logistic.py — Logistic Regression with ElasticNet regularization

Hyperparameters are tuned via Bayesian search using Optuna. All data
transformation is delegated to preprocessing.py; no transformation logic
lives here.

Outputs written to data/:
  lr_model.pkl         — fitted LogisticRegression (joblib)
  lr_results.json      — metrics at both thresholds, hyperparams, coefficients
  lr_coefficients.csv  — feature coefficient table sorted by |coef| descending
  optuna_study.pkl     — full Optuna study object for dashboard trial history
"""

import json
import logging
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from preprocessing import (
    DATA_PATH,
    build_feature_matrix,
    get_class_weights,
    get_cv_splitter,
    load_significant_features,
    split_data,
)

LR_MODEL_PATH   = "data/lr_model.pkl"
LR_RESULTS_PATH = "data/lr_results.json"
LR_COEF_PATH    = "data/lr_coefficients.csv"
STUDY_PATH      = "data/optuna_study.pkl"

N_TRIALS = 100


# ── Optuna objective ───────────────────────────────────────────────────────────

def define_objective(trial, X_train, y_train, cv, class_weights):
    """Optuna objective: return mean CV AUC-ROC for a set of sampled hyperparams."""
    C         = trial.suggest_float("C", 0.001, 100.0, log=True)
    l1_ratio  = trial.suggest_float("l1_ratio", 0.0, 1.0)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=2000,
        class_weight=class_weights,
        random_state=42,
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


# ── Threshold optimization ─────────────────────────────────────────────────────

def _find_optimal_threshold(
    y_train: np.ndarray,
    train_prob: np.ndarray,
    sensitivity_weight: float = 0.6,
) -> float:
    """Find the threshold maximizing weighted Youden's J statistic.

    Objective: maximize (sensitivity_weight * sensitivity) +
               ((1 - sensitivity_weight) * specificity)

    This balances both classes while giving slight emphasis to sensitivity
    (avoiding missed strokes), avoiding the degenerate case where a very
    low threshold classifies everything as positive to maximize recall.

    sensitivity_weight=0.6 means sensitivity is weighted 60%,
    specificity 40%.
    """
    fpr, tpr, thresholds = roc_curve(y_train, train_prob)
    specificity = 1.0 - fpr

    weighted_j = (sensitivity_weight * tpr) + ((1 - sensitivity_weight) * specificity)
    best_idx = int(np.argmax(weighted_j))
    optimal_threshold = float(thresholds[best_idx])

    print(
        f"Optimal threshold: {optimal_threshold:.4f}  "
        f"(sensitivity={tpr[best_idx]:.3f}, specificity={specificity[best_idx]:.3f}, "
        f"weighted J={weighted_j[best_idx]:.3f})"
    )
    return optimal_threshold


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str,
) -> dict:
    """Compute classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = precision_score(y_true, y_pred, zero_division=0)
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_true, y_prob)

    print(f"\n  [{label}]  threshold = {threshold:.4f}")
    print(f"    Accuracy    : {accuracy:.4f}")
    print(f"    Sensitivity : {sensitivity:.4f}  (recall / TPR)")
    print(f"    Specificity : {specificity:.4f}  (TNR)")
    print(f"    Precision   : {precision:.4f}  (PPV)")
    print(f"    NPV         : {npv:.4f}")
    print(f"    F1          : {f1:.4f}")
    print(f"    AUC-ROC     : {roc_auc:.4f}")
    print(f"    Confusion matrix (TN FP / FN TP): [{tn} {fp} / {fn} {tp}]")

    return {
        "threshold":   threshold,
        "accuracy":    round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision":   round(precision, 4),
        "npv":         round(npv, 4),
        "f1":          round(f1, 4),
        "roc_auc":     round(roc_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load and preprocess ─────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    features = load_significant_features()
    print()

    X, y = build_feature_matrix(df, features)
    print()

    X_train, X_test, y_train, y_test = split_data(X, y)
    print()

    class_weights = get_class_weights(y_train)
    cv            = get_cv_splitter(n_splits=5)
    print()

    # ── 2. Optuna hyperparameter search ────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"Running Optuna study ({N_TRIALS} trials) ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: define_objective(trial, X_train, y_train, cv, class_weights),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_auc = study.best_value
    print(f"\nBest CV AUC-ROC : {best_cv_auc:.4f}")
    print(f"Best params     : C={best_params['C']:.5f}, l1_ratio={best_params['l1_ratio']:.4f}")

    # ── 3. Refit best model on full training set ───────────────────────────────
    print("\nRefitting best model on full training set ...")
    best_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=2000,
        class_weight=class_weights,
        random_state=42,
    )
    best_model.fit(X_train, y_train)

    # ── 4. Threshold optimization on training set ──────────────────────────────
    train_prob        = best_model.predict_proba(X_train)[:, 1]
    optimal_threshold = _find_optimal_threshold(y_train.values, train_prob)
    default_threshold = 0.5
    print(f"\nThresholds - default: {default_threshold}, optimal: {optimal_threshold:.4f}")

    # ── 5. Evaluate on test set ────────────────────────────────────────────────
    test_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nTest-set evaluation:")
    metrics_default = _evaluate_at_threshold(
        y_test.values, test_prob, default_threshold, "Default threshold = 0.50"
    )
    metrics_optimal = _evaluate_at_threshold(
        y_test.values, test_prob, optimal_threshold,
        f"Optimal threshold = {optimal_threshold:.4f}"
    )

    # ── 6. Coefficients ────────────────────────────────────────────────────────
    feature_names = list(X_train.columns)
    coefs         = best_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature":      feature_names,
        "coefficient":  coefs,
        "odds_ratio":   np.exp(coefs),
        "abs_coef":     np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    zeroed = coef_df[coef_df["coefficient"] == 0.0]["feature"].tolist()
    retained = coef_df[coef_df["coefficient"] != 0.0]["feature"].tolist()

    print(f"\nCoefficients - {len(retained)} retained, {len(zeroed)} zeroed by L1:")
    if zeroed:
        print(f"  Zeroed: {', '.join(zeroed)}")
    print(coef_df[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    joblib.dump(best_model, LR_MODEL_PATH)
    print(f"\nModel saved to {LR_MODEL_PATH}")

    joblib.dump(study, STUDY_PATH)
    print(f"Optuna study saved to {STUDY_PATH}")

    coef_df.to_csv(LR_COEF_PATH, index=False)
    print(f"Coefficients saved to {LR_COEF_PATH}")

    results = {
        "best_hyperparams": {
            "C":        round(best_params["C"], 6),
            "l1_ratio": round(best_params["l1_ratio"], 6),
        },
        "best_cv_auc":         round(best_cv_auc, 4),
        "default_threshold":   default_threshold,
        "optimal_threshold":   round(optimal_threshold, 4),
        "metrics_default":     metrics_default,
        "metrics_optimal":     metrics_optimal,
        "feature_coefficients": coef_df[
            ["feature", "coefficient", "odds_ratio", "abs_coef"]
        ].round(6).to_dict(orient="records"),
    }
    with open(LR_RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {LR_RESULTS_PATH}")

    # ── 8. Plain-English summary ───────────────────────────────────────────────
    mo = metrics_optimal
    md = metrics_default

    top_features = coef_df[coef_df["coefficient"] != 0.0].head(3)["feature"].tolist()
    top_str = ", ".join(top_features)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"The logistic regression model achieved a cross-validated AUC-ROC of "
        f"{best_cv_auc:.3f} (best C={best_params['C']:.4f}, "
        f"l1_ratio={best_params['l1_ratio']:.3f}). "
        f"At the default 0.5 threshold, test-set sensitivity was "
        f"{md['sensitivity']:.3f} with specificity {md['specificity']:.3f} "
        f"and AUC-ROC {md['roc_auc']:.3f}. "
        f"The optimal threshold ({optimal_threshold:.3f}) was selected by maximizing a "
        f"weighted Youden's J statistic (60% sensitivity, 40% specificity weight), "
        f"reflecting the clinical priority of minimizing missed stroke cases while "
        f"maintaining meaningful specificity. This raised recall to {mo['sensitivity']:.3f} "
        f"at the cost of specificity ({mo['specificity']:.3f}) and precision "
        f"({mo['precision']:.3f}), reflecting the class imbalance inherent in a "
        f"~5% stroke prevalence dataset. "
        f"The strongest predictors by absolute coefficient were {top_str}. "
        + (
            f"L1 regularization zeroed out {len(zeroed)} feature(s) "
            f"({', '.join(zeroed)}), leaving {len(retained)} active."
            if zeroed else
            f"No features were zeroed out; all {len(retained)} were retained by the model."
        )
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
