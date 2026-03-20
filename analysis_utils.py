import pandas as pd
import numpy as np

CATEGORICAL = ["gender", "hypertension", "heart_disease", "ever_married",
               "work_type", "Residence_type", "smoking_status"]
NUMERIC = ["age", "avg_glucose_level", "bmi"]
ALL_FEATURES = CATEGORICAL + NUMERIC

ENCODE_MAP = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {"children": 0, "Never_worked": 1, "Govt_job": 2, "Private": 3, "Self-employed": 4},
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {"never smoked": 0, "formerly smoked": 1, "smokes": 2},
}


def stroke_rate_by_category(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    result = (
        df.groupby(feature)["stroke"]
        .agg(stroke_rate="mean", count="count")
        .reset_index()
    )
    result["stroke_pct"] = result["stroke_rate"] * 100
    return result


def stroke_rate_by_numeric(df: pd.DataFrame, feature: str, n_bins: int = 10) -> pd.DataFrame:
    binned = pd.cut(df[feature], bins=n_bins)
    result = (
        df.groupby(binned, observed=True)["stroke"]
        .agg(stroke_rate="mean", count="count")
        .reset_index()
    )
    result.columns = [feature, "stroke_rate", "count"]
    result["stroke_pct"] = result["stroke_rate"] * 100
    result["bin_label"] = result[feature].astype(str)
    return result


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """filters: {feature: ("eq", value) | ("range", (lo, hi))}"""
    mask = pd.Series(True, index=df.index)
    for feat, (ftype, val) in filters.items():
        if ftype == "eq":
            mask &= df[feat] == val
        else:
            mask &= (df[feat] >= val[0]) & (df[feat] <= val[1])
    return df[mask]


def encoded_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df[CATEGORICAL + NUMERIC + ["stroke"]].copy()
    for col, mapping in ENCODE_MAP.items():
        df_enc[col] = df_enc[col].map(mapping)
    return df_enc
