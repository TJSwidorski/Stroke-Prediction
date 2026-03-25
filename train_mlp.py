"""
train_mlp.py — Multi-Layer Perceptron training with four architecture configurations

Hyperparameter choices are fixed per CONFIG; no search is performed here.
All data transformation is delegated to preprocessing.py; no transformation
logic lives in this file.

Outputs written to data/:
  mlp_{name}_model/           — SavedModel format (one directory per config)
  mlp_{name}_history.json     — loss, val_loss, auc, val_auc per epoch
  mlp_{name}_results.json     — metrics at both thresholds, optimal_threshold, config spec
  mlp_attention_weights.json  — attention weights from Config D (feature → weight, desc.)
  mlp_comparison.csv          — one row per config, all test-set metrics at optimal threshold
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import (
    DATA_PATH,
    build_feature_matrix,
    get_class_weights,
    load_significant_features,
    split_data,
)

# ── Architecture configurations ────────────────────────────────────────────────

CONFIGS = [
    {
        "name":           "Shallow Wide",
        "layers":         [128],
        "dropout_rate":   0.0,
        "l2_reg":         0.0,
        "use_batch_norm": False,
        "use_leaky_relu": False,
        "use_attention":  False,
        "learning_rate":  0.001,
        "batch_size":     32,
    },
    {
        "name":           "Medium Dropout",
        "layers":         [64, 32],
        "dropout_rate":   0.3,
        "l2_reg":         0.0,
        "use_batch_norm": False,
        "use_leaky_relu": False,
        "use_attention":  False,
        "learning_rate":  0.001,
        "batch_size":     32,
    },
    {
        "name":           "Deep Regularized",
        "layers":         [128, 64, 32],
        "dropout_rate":   0.3,
        "l2_reg":         0.001,
        "use_batch_norm": False,
        "use_leaky_relu": False,
        "use_attention":  False,
        "learning_rate":  0.001,
        "batch_size":     32,
    },
    {
        "name":           "Attention Weighted",
        "layers":         [64, 32],
        "dropout_rate":   0.2,
        "l2_reg":         0.0,
        "use_batch_norm": False,
        "use_leaky_relu": False,
        "use_attention":  True,
        "learning_rate":  0.001,
        "batch_size":     32,
    },
]


def _safe_name(name: str) -> str:
    """Convert config name to a filesystem-safe snake_case string."""
    return name.lower().replace(" ", "_")


# ── Model construction ─────────────────────────────────────────────────────────

def build_model(config: dict, input_dim: int) -> tf.keras.Model:
    """Construct and compile a Keras model from a config dict.

    If use_attention=True, a softmax Dense layer named "attention_weights"
    is multiplied element-wise with the raw inputs before the hidden layers,
    so per-feature importance can be extracted after training.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="inputs")

    if config["use_attention"]:
        attn = tf.keras.layers.Dense(
            input_dim, activation="softmax", name="attention_weights"
        )(inputs)
        x = tf.keras.layers.Multiply()([inputs, attn])
    else:
        x = inputs

    regularizer = (
        tf.keras.regularizers.l2(config["l2_reg"]) if config["l2_reg"] > 0 else None
    )

    for units in config["layers"]:
        x = tf.keras.layers.Dense(units, kernel_regularizer=regularizer)(x)
        if config["use_batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        if config["use_leaky_relu"]:
            x = tf.keras.layers.LeakyReLU(0.01)(x)
        else:
            x = tf.keras.layers.ReLU()(x)
        if config["dropout_rate"] > 0:
            x = tf.keras.layers.Dropout(config["dropout_rate"])(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name=_safe_name(config["name"])
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train_config(
    config: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_weights: dict,
) -> tuple:
    """Build, fit, and return (model, history) for one architecture config."""
    model = build_model(config, input_dim=X_train.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=15,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            patience=7,
            factor=0.5,
            mode="max",
        ),
    ]

    history = model.fit(
        X_train.values,
        y_train.values,
        epochs=150,
        batch_size=config["batch_size"],
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str,
) -> dict:
    """Compute and print classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
        "threshold":        threshold,
        "accuracy":         round(accuracy, 4),
        "sensitivity":      round(sensitivity, 4),
        "specificity":      round(specificity, 4),
        "precision":        round(precision, 4),
        "npv":              round(npv, 4),
        "f1":               round(f1, 4),
        "roc_auc":          round(roc_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _find_optimal_threshold(
    y_train: np.ndarray,
    train_prob: np.ndarray,
    target_sensitivity: float = 0.85,
) -> float:
    """Return the lowest threshold on training ROC achieving target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_train, train_prob)
    candidates = [(t, s) for t, s in zip(thresholds, tpr) if s >= target_sensitivity]
    if candidates:
        return float(min(candidates, key=lambda x: x[0])[0])
    optimal = float(thresholds[np.argmax(tpr)])
    print(
        f"\nWARNING: No threshold achieves sensitivity >= {target_sensitivity:.0%}. "
        f"Using threshold with highest sensitivity ({tpr.max():.4f}) instead."
    )
    return optimal


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load and preprocess ─────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    features = load_significant_features()
    print()

    X, y = build_feature_matrix(df, features)
    feature_names = list(X.columns)
    print()

    X_train, X_test, y_train, y_test = split_data(X, y)
    print()

    class_weights = get_class_weights(y_train)
    print()

    os.makedirs("data", exist_ok=True)

    # ── 2. Train each config ───────────────────────────────────────────────────
    comparison_rows      = []
    attention_weights_out = None

    for config in CONFIGS:
        safe = _safe_name(config["name"])
        print("\n" + "=" * 70)
        print(f"CONFIG: {config['name']}")
        print("=" * 70)

        model, history = train_config(
            config, X_train, X_test, y_train, y_test, class_weights
        )

        # Threshold optimization on training predictions
        train_prob        = model.predict(X_train.values, verbose=0).ravel()
        optimal_threshold = _find_optimal_threshold(y_train.values, train_prob)
        default_threshold = 0.5
        print(
            f"\nThresholds — default: {default_threshold}, "
            f"optimal: {optimal_threshold:.4f}"
        )

        # Test-set evaluation at both thresholds
        test_prob = model.predict(X_test.values, verbose=0).ravel()
        print("\nTest-set evaluation:")
        metrics_default = _evaluate_at_threshold(
            y_test.values, test_prob, default_threshold, "Default threshold = 0.50"
        )
        metrics_optimal = _evaluate_at_threshold(
            y_test.values, test_prob, optimal_threshold,
            f"Optimal threshold = {optimal_threshold:.4f}",
        )

        # Attention weight extraction (Config D only)
        if config["use_attention"]:
            attn_extractor = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer("attention_weights").output,
            )
            attn_matrix           = attn_extractor.predict(X_train.values, verbose=0)
            avg_weights           = attn_matrix.mean(axis=0)
            attention_weights_out = dict(
                sorted(
                    {n: float(w) for n, w in zip(feature_names, avg_weights)}.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            )
            print("\nAttention weights (top 5 features):")
            for feat, w in list(attention_weights_out.items())[:5]:
                print(f"  {feat}: {w:.4f}")

        # Save model
        model_path = f"data/mlp_{safe}_model"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Save history — keep the four core series the dashboard needs,
        # plus any extra keys Keras records (lr, val_accuracy, etc.)
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        history_path = f"data/mlp_{safe}_history.json"
        with open(history_path, "w") as fh:
            json.dump(hist_dict, fh, indent=2)
        print(f"History saved to {history_path}")

        # Save per-config results
        results = {
            "config":            config,
            "default_threshold": default_threshold,
            "optimal_threshold": round(optimal_threshold, 4),
            "metrics_default":   metrics_default,
            "metrics_optimal":   metrics_optimal,
        }
        results_path = f"data/mlp_{safe}_results.json"
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {results_path}")

        # Accumulate comparison row (metrics at optimal threshold)
        mo = metrics_optimal
        comparison_rows.append({
            "Config":            config["name"],
            "Layers":            str(config["layers"]),
            "Dropout":           config["dropout_rate"],
            "L2":                config["l2_reg"],
            "Attention":         config["use_attention"],
            "Optimal_Threshold": round(optimal_threshold, 4),
            "AUC_ROC":           mo["roc_auc"],
            "Sensitivity":       mo["sensitivity"],
            "Specificity":       mo["specificity"],
            "Precision":         mo["precision"],
            "NPV":               mo["npv"],
            "F1":                mo["f1"],
            "Accuracy":          mo["accuracy"],
        })

    # ── 3. Save attention weights ──────────────────────────────────────────────
    if attention_weights_out is not None:
        attn_path = "data/mlp_attention_weights.json"
        with open(attn_path, "w") as fh:
            json.dump(attention_weights_out, fh, indent=2)
        print(f"\nAttention weights saved to {attn_path}")

    # ── 4. Save comparison CSV ─────────────────────────────────────────────────
    comp_df   = pd.DataFrame(comparison_rows)
    comp_path = "data/mlp_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Comparison table saved to {comp_path}")

    # ── 5. Print final comparison table ───────────────────────────────────────
    display_cols = [
        "Config", "AUC_ROC", "Sensitivity", "Specificity", "F1", "Optimal_Threshold"
    ]
    print("\n" + "=" * 70)
    print("COMPARISON — all configs at optimal threshold")
    print("=" * 70)
    print(comp_df[display_cols].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
