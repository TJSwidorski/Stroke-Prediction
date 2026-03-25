"""
preprocessing.py — shared feature engineering for model training scripts

Imported by logistic regression and neural network training scripts.
Does not run any training itself.

Outputs written to data/:
  scaler.pkl          — fitted StandardScaler (reused by inference scripts)
  feature_columns.json — ordered column list after one-hot encoding
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH    = "data/stroke_data_clean.csv"
RESULTS_PATH = "data/phase2_hypothesis_results.csv"
SCALER_PATH  = "data/scaler.pkl"
COLUMNS_PATH = "data/feature_columns.json"

NUMERIC_FEATURES  = ["age", "bmi", "avg_glucose_level"]
OHE_FEATURES      = ["work_type", "smoking_status"]
BINARY_FEATURES   = ["hypertension", "heart_disease", "ever_married", "Residence_type"]
DROP_COLUMNS      = ["id", "stroke", "age_outlier", "glucose_outlier", "bmi_outlier"]

BINARY_MAP = {
    "ever_married":   {"No": 0, "Yes": 1},
    "Residence_type": {"Rural": 0, "Urban": 1},
}


# ── 1. Feature selection ───────────────────────────────────────────────────────

def load_significant_features() -> list[str]:
    """Read phase2_hypothesis_results.csv and return features where Sig. != 'ns'."""
    results = pd.read_csv(RESULTS_PATH)
    sig_features = results[results["Sig."] != "ns"]["Feature"].tolist()
    print(f"Significant features selected ({len(sig_features)}):")
    for f in sig_features:
        print(f"  {f}")
    return sig_features


# ── 2. Feature matrix ──────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X (feature matrix) and y (stroke target) from the cleaned dataframe.

    Preprocessing applied:
    - Binary map: ever_married (No/Yes → 0/1), Residence_type (Rural/Urban → 0/1)
    - One-hot encode: work_type, smoking_status (all dummies, drop_first=False)
    - StandardScaler: age, bmi, avg_glucose_level (scaler saved to data/scaler.pkl)
    - Drops: id, stroke, age_outlier, glucose_outlier, bmi_outlier
    - Column list saved to data/feature_columns.json

    Returns X as a DataFrame with column names preserved, y as a Series.
    """
    working = df.copy()

    # Apply binary mappings for string-valued binary features
    for col, mapping in BINARY_MAP.items():
        if col in working.columns:
            working[col] = working[col].map(mapping)

    # One-hot encode multi-level categoricals present in the feature list
    ohe_cols = [c for c in OHE_FEATURES if c in features and c in working.columns]
    if ohe_cols:
        working = pd.get_dummies(working, columns=ohe_cols, drop_first=False)

    # Drop columns not used in modelling
    working = working.drop(
        columns=[c for c in DROP_COLUMNS if c in working.columns]
    )

    # Drop any remaining categorical columns not in our feature set
    # (e.g. gender if not significant — keep only columns derived from `features`
    #  plus the OHE expansions we just created)
    base_keep = set(features) - set(OHE_FEATURES)
    ohe_keep  = {c for c in working.columns if any(c.startswith(f + "_") for f in ohe_cols)}
    keep_cols = [c for c in working.columns if c in base_keep or c in ohe_keep]
    working = working[keep_cols]

    # Scale numeric features
    numeric_present = [c for c in NUMERIC_FEATURES if c in working.columns]
    scaler = StandardScaler()
    working[numeric_present] = scaler.fit_transform(working[numeric_present])
    joblib.dump(scaler, SCALER_PATH)
    print(f"StandardScaler fitted on {numeric_present} and saved to {SCALER_PATH}")

    # Persist column order for downstream scripts
    feature_columns = list(working.columns)
    with open(COLUMNS_PATH, "w") as fh:
        json.dump(feature_columns, fh, indent=2)
    print(f"Feature columns ({len(feature_columns)}) saved to {COLUMNS_PATH}")

    # Cast to float32 — pd.get_dummies produces bool columns in pandas ≥ 2.0,
    # which sklearn handles silently but Keras rejects with "Invalid dtype: object".
    working = working.astype(np.float32)

    y = df["stroke"].astype(int)
    return working, y


# ── 3. Train/test split ────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Prints class distribution for both splits to confirm stratification held.
    Returns X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    for label, y_split in [("Train", y_train), ("Test", y_test)]:
        counts = y_split.value_counts().sort_index()
        pct    = y_split.value_counts(normalize=True).sort_index() * 100
        print(
            f"{label} split (n={len(y_split):,}): "
            f"no stroke={counts[0]:,} ({pct[0]:.1f}%), "
            f"stroke={counts[1]:,} ({pct[1]:.1f}%)"
        )

    return X_train, X_test, y_train, y_test


# ── 4. Class weights ───────────────────────────────────────────────────────────

def get_class_weights(y_train: pd.Series) -> dict[int, float]:
    """Compute balanced class weights to handle stroke class imbalance.

    Returns {0: w0, 1: w1}. Prints the computed weights.
    """
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(
        f"Class weights - no stroke: {weight_dict[0]:.4f}, "
        f"stroke: {weight_dict[1]:.4f} "
        f"(ratio 1:{weight_dict[1] / weight_dict[0]:.1f})"
    )
    return weight_dict


# ── 5. CV splitter ─────────────────────────────────────────────────────────────

def get_cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    """Return a StratifiedKFold splitter with shuffle and fixed random state."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ── CLI sanity check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    features = load_significant_features()
    print()

    X, y = build_feature_matrix(df, features)
    print()

    X_train, X_test, y_train, y_test = split_data(X, y)
    print()

    weights = get_class_weights(y_train)
    print()

    ratio = weights[1] / weights[0]
    print("=" * 50)
    print("Preprocessing summary")
    print("=" * 50)
    print(f"  Features after encoding : {X.shape[1]}")
    print(f"  Train size              : {len(X_train):,}")
    print(f"  Test size               : {len(X_test):,}")
    print(f"  Class weight ratio (1:0): {ratio:.2f}")
    print("=" * 50)
