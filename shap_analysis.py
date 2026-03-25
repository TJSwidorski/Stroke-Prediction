"""
shap_analysis.py — SHAP feature importance for logistic regression and best MLP

Requires train_logistic.py and train_mlp.py outputs in data/.

Outputs written to data/:
  shap_lr_values.npy               — raw SHAP values (n_test × n_features)
  shap_lr_feature_importance.csv   — feature, mean_abs_shap (descending)
  shap_lr_summary.png              — SHAP beeswarm plot
  shap_lr_bar.png                  — SHAP mean-absolute bar plot
  shap_mlp_values.npy              — raw SHAP values for best MLP config
  shap_mlp_feature_importance.csv  — feature, mean_abs_shap (descending)
  shap_mlp_summary.png
  shap_mlp_bar.png
  feature_importance_comparison.csv — normalized comparison across all methods
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from preprocessing import (
    DATA_PATH,
    build_feature_matrix,
    load_significant_features,
    split_data,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

LR_MODEL_PATH    = "data/lr_model.pkl"
LR_COEF_PATH     = "data/lr_coefficients.csv"
MLP_COMPARE_PATH = "data/mlp_comparison.csv"
ATTN_PATH        = "data/mlp_attention_weights.json"


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


# ── SHAP helpers ───────────────────────────────────────────────────────────────

def _to_2d(raw) -> np.ndarray:
    """Coerce any SHAP output form to a 2D float32 array (n_samples, n_features).

    Handles: list-wrapped arrays (multi-output explainers), Explanation objects,
    and 3D tensors from some DeepExplainer versions.
    """
    if isinstance(raw, list):
        # Binary classification explainers sometimes return [neg_class, pos_class];
        # single-output explainers return a one-element list.
        vals = raw[1] if len(raw) == 2 else raw[0]
    else:
        vals = raw
    if hasattr(vals, "values"):         # shap.Explanation object
        vals = vals.values
    arr = np.asarray(vals, dtype=np.float32)
    if arr.ndim == 3:                   # (n_samples, n_features, n_outputs)
        arr = arr[:, :, 0]
    return arr


def _save_plots(
    shap_values: np.ndarray,
    X_np: np.ndarray,
    feature_names: list,
    prefix: str,
) -> None:
    """Write beeswarm summary and mean-absolute bar plots to data/."""
    shap.summary_plot(shap_values, X_np, feature_names=feature_names, show=False)
    path = f"data/{prefix}_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")

    shap.summary_plot(
        shap_values, X_np, feature_names=feature_names, plot_type="bar", show=False
    )
    path = f"data/{prefix}_bar.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")


def _save_importance(
    shap_values: np.ndarray,
    feature_names: list,
    npy_path: str,
    csv_path: str,
) -> pd.DataFrame:
    """Persist raw SHAP array and mean-absolute feature importance CSV."""
    np.save(npy_path, shap_values)
    print(f"  SHAP values → {npy_path}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(csv_path, index=False)
    print(f"  Feature importance → {csv_path}")
    return imp_df


# ── Section 1 — Logistic Regression SHAP ──────────────────────────────────────

def shap_logistic(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
) -> None:
    print("\n" + "=" * 60)
    print("SECTION 1 — Logistic Regression SHAP")
    print("=" * 60)

    lr_model = joblib.load(LR_MODEL_PATH)
    print(f"Loaded {LR_MODEL_PATH}")

    explainer = shap.LinearExplainer(lr_model, X_train.values)
    raw       = explainer.shap_values(X_test.values)
    sv        = _to_2d(raw)
    print(f"  SHAP values shape: {sv.shape}")

    imp = _save_importance(sv, feature_names,
                           "data/shap_lr_values.npy",
                           "data/shap_lr_feature_importance.csv")
    _save_plots(sv, X_test.values, feature_names, "shap_lr")

    print("\nTop 5 features (LR SHAP):")
    print(imp.head(5).to_string(index=False))


# ── Section 2 — Best MLP SHAP ─────────────────────────────────────────────────

def shap_mlp(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
) -> None:
    print("\n" + "=" * 60)
    print("SECTION 2 — Best MLP SHAP")
    print("=" * 60)

    comp     = pd.read_csv(MLP_COMPARE_PATH)
    best_row = comp.loc[comp["AUC_ROC"].idxmax()]
    best_cfg = best_row["Config"]
    safe     = _safe_name(best_cfg)
    print(f"Best config: {best_cfg}  (AUC-ROC = {best_row['AUC_ROC']:.4f})")

    model_path = f"data/mlp_{safe}_model.keras"
    model      = tf.keras.models.load_model(model_path)
    print(f"Loaded {model_path}")

    X_train_np = X_train.values
    X_test_np  = X_test.values
    background = shap.sample(X_train_np, 100, random_state=42)

    sv     = None
    method = None

    try:
        explainer = shap.DeepExplainer(model, background)
        raw       = explainer.shap_values(X_test_np)
        sv        = _to_2d(raw)
        method    = "DeepExplainer"
        print(f"  Used DeepExplainer. SHAP values shape: {sv.shape}")
    except Exception as exc:
        print(f"  DeepExplainer failed ({exc})")
        print("  Falling back to KernelExplainer (100-sample background) ...")

        def _predict(x):
            return model.predict(x, verbose=0).ravel()

        explainer = shap.KernelExplainer(_predict, background)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = explainer.shap_values(X_test_np, nsamples=100)
        sv     = _to_2d(raw)
        method = "KernelExplainer"
        print(f"  Used KernelExplainer. SHAP values shape: {sv.shape}")

    imp = _save_importance(sv, feature_names,
                           "data/shap_mlp_values.npy",
                           "data/shap_mlp_feature_importance.csv")
    _save_plots(sv, X_test_np, feature_names, "shap_mlp")

    print(f"\nTop 5 features (MLP SHAP via {method}):")
    print(imp.head(5).to_string(index=False))


# ── Section 3 — Feature importance comparison ─────────────────────────────────

def compare_importance(feature_names: list) -> None:
    print("\n" + "=" * 60)
    print("SECTION 3 — Feature Importance Comparison")
    print("=" * 60)

    lr_coef  = pd.read_csv(LR_COEF_PATH)[["feature", "abs_coef"]]
    shap_lr  = (pd.read_csv("data/shap_lr_feature_importance.csv")
                  [["feature", "mean_abs_shap"]]
                  .rename(columns={"mean_abs_shap": "shap_lr"}))
    shap_mlp = (pd.read_csv("data/shap_mlp_feature_importance.csv")
                   [["feature", "mean_abs_shap"]]
                   .rename(columns={"mean_abs_shap": "shap_mlp"}))

    with open(ATTN_PATH) as fh:
        attn_raw = json.load(fh)
    attn_df = pd.DataFrame(
        list(attn_raw.items()), columns=["feature", "attention_weight"]
    )

    # Merge all sources onto the canonical feature list as the spine
    out = pd.DataFrame({"feature": feature_names})
    out = out.merge(lr_coef,  on="feature", how="left")
    out = out.merge(shap_lr,  on="feature", how="left")
    out = out.merge(shap_mlp, on="feature", how="left")
    out = out.merge(attn_df,  on="feature", how="left")
    out = out.rename(columns={"abs_coef": "lr_coef_abs"})

    # Min-max normalize each importance column to [0, 1]
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return s if hi == lo else (s - lo) / (hi - lo)

    for col in ["lr_coef_abs", "shap_lr", "shap_mlp", "attention_weight"]:
        out[col] = _norm(out[col])

    out.to_csv("data/feature_importance_comparison.csv", index=False)
    print("Saved data/feature_importance_comparison.csv")

    # Print top-5 rankings side by side
    methods = {
        "LR Coef |β|": "lr_coef_abs",
        "SHAP (LR)":   "shap_lr",
        "SHAP (MLP)":  "shap_mlp",
        "Attention":   "attention_weight",
    }
    top5 = {
        label: out.dropna(subset=[col]).sort_values(col, ascending=False)["feature"].head(5).tolist()
        for label, col in methods.items()
    }

    rows = [
        {"Rank": i + 1, **{label: (top5[label][i] if i < len(top5[label]) else "—")
                           for label in methods}}
        for i in range(5)
    ]
    print("\nTop 5 features by each importance method (normalized 0–1):")
    print(pd.DataFrame(rows).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    features      = load_significant_features()
    X, y          = build_feature_matrix(df, features)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    os.makedirs("data", exist_ok=True)

    shap_logistic(X_train, X_test, feature_names)
    shap_mlp(X_train, X_test, feature_names)
    compare_importance(feature_names)

    print("\n" + "=" * 60)
    print("SHAP analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
