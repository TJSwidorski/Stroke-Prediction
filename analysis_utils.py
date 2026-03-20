import pandas as pd
import numpy as np
from scipy import stats

CATEGORICAL = ["gender", "hypertension", "heart_disease", "ever_married",
               "work_type", "Residence_type", "smoking_status"]
NUMERIC = ["age", "avg_glucose_level", "bmi"]
ALL_FEATURES = CATEGORICAL + NUMERIC

DEFAULT_BIN_WIDTHS = {"age": 10, "avg_glucose_level": 50, "bmi": 5}

BIN_WIDTH_OPTIONS = {
    "age":               [1, 2, 3, 4, 5, 10, 15, 20],
    "avg_glucose_level": [1, 5, 10, 15, 20, 25, 50, 75, 100],
    "bmi":               [1, 2, 3, 4, 5, 10, 15, 20],
}

ENCODE_MAP = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {"children": 0, "Never_worked": 1, "Govt_job": 2, "Private": 3, "Self-employed": 4},
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {"never smoked": 0, "formerly smoked": 1, "smokes": 2},
}


# ── Agresti-Coull interval ─────────────────────────────────────────────────────

def agresti_coull_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% (or 1-alpha) Agresti-Coull confidence interval for a proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    n_tilde = n + z ** 2
    p_tilde = (successes + z ** 2 / 2) / n_tilde
    margin = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return float(np.clip(p_tilde - margin, 0, 1)), float(np.clip(p_tilde + margin, 0, 1))


# ── Proportion z-test ──────────────────────────────────────────────────────────

def proportion_z_test(successes: int, n: int, p0: float) -> tuple[float, float]:
    """Two-sided z-test for a single proportion against null p0.
    Returns (z_statistic, p_value)."""
    if n == 0:
        return np.nan, np.nan
    p_hat = successes / n
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return float(z), float(p_value)


def sig_stars(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ── Binning helper ─────────────────────────────────────────────────────────────

def _make_bins(series: pd.Series, bin_width: float) -> np.ndarray:
    """Compute bin edges aligned to multiples of bin_width."""
    lo = np.floor(series.min() / bin_width) * bin_width
    hi = np.ceil(series.max() / bin_width) * bin_width
    return np.arange(lo, hi + bin_width * 0.5, bin_width)


def _bin_label(interval, bin_width: float) -> str:
    lo, hi = interval.left, interval.right
    if bin_width >= 1:
        return f"{int(lo)}–{int(hi)}"
    return f"{lo:.1f}–{hi:.1f}"


# ── Per-feature stats (rate + AC CI + z-test) ──────────────────────────────────

def feature_stroke_stats(df: pd.DataFrame, feature: str,
                         bin_width: float | None = None,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Stroke rate, Agresti-Coull CI, and proportion z-test for each
    value (categorical) or bin (numeric) of a feature.

    bin_width applies only to numeric features; defaults to DEFAULT_BIN_WIDTHS.
    """
    p0 = df["stroke"].mean()

    if feature in CATEGORICAL:
        groups = (
            df.groupby(feature, observed=True)["stroke"]
            .agg(strokes="sum", n="count")
            .reset_index()
        )
        groups["label"] = groups[feature].astype(str)
    else:
        bw = bin_width if bin_width is not None else DEFAULT_BIN_WIDTHS[feature]
        bins = _make_bins(df[feature], bw)
        cut = pd.cut(df[feature], bins=bins, include_lowest=True)
        groups = (
            df.assign(_bin=cut)
            .groupby("_bin", observed=True)["stroke"]
            .agg(strokes="sum", n="count")
            .reset_index()
        )
        groups.columns = ["_bin", "strokes", "n"]
        groups["label"] = groups["_bin"].apply(lambda iv: _bin_label(iv, bw))

    records = []
    for _, row in groups.iterrows():
        x, n = int(row["strokes"]), int(row["n"])
        rate = x / n if n > 0 else np.nan
        lo, hi = agresti_coull_ci(x, n, alpha) if n > 0 else (np.nan, np.nan)
        z, p_val = proportion_z_test(x, n, p0) if n > 0 else (np.nan, np.nan)
        sig = not np.isnan(p_val) and p_val < alpha

        records.append({
            "feature": feature,
            "label": row["label"],
            "n": n,
            "strokes": x,
            "rate": rate,
            "ci_lower": lo,
            "ci_upper": hi,
            "z_stat": z,
            "p_value": p_val,
            "sig": sig_stars(p_val),
            "significant": sig,
            "direction": ("Higher ▲" if rate > p0 else "Lower ▼") if sig else "—",
        })

    return pd.DataFrame(records)


def all_features_stats(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run feature_stroke_stats for every feature using default bin widths."""
    return pd.concat(
        [feature_stroke_stats(df, f, alpha=alpha) for f in ALL_FEATURES],
        ignore_index=True,
    )


# ── Misc helpers ───────────────────────────────────────────────────────────────

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
