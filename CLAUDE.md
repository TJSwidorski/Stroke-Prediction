# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a clinical medical research project that applies machine learning to the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`fedesoriano/stroke-prediction-dataset` on Kaggle). The goal is to identify which cardiovascular and lifestyle risk factors predict strokes and build a model to flag high-risk patients.

## Setup

Install dependencies:
```bash
pip install kagglehub[pandas-datasets] python-dotenv scikit-learn pandas numpy scipy streamlit plotly
```

Kaggle credentials go in a `.env` file (excluded from git):
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Pipeline

Run scripts in order:

1. **`retrieve_data.py`** — downloads dataset from Kaggle, saves to `data/stroke_data.csv`
2. **`feature_engineering.py`** — KNN imputes missing BMI and Unknown smoking status, saves to `data/stroke_data_clean.csv` and `data/data_quality_report.txt`
3. **`hypothesis_testing.py`** — Phase 2 group comparison analysis, prints results to console and saves to `data/phase2_hypothesis_results.csv`
4. **`dashboard.py`** — Streamlit interactive analysis dashboard (reads `data/stroke_data_clean.csv`)

```bash
python retrieve_data.py
python feature_engineering.py
python hypothesis_testing.py
streamlit run dashboard.py
```

`data/` is gitignored — raw and cleaned CSVs are not committed.

## Data

Key columns: `age`, `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke` (target).

Known data quality issues (handled in `feature_engineering.py`):
- `bmi`: NaN values → filled via KNN (k=5)
- `smoking_status`: `"Unknown"` entries → filled via KNN (k=5) using all other encoded features

`feature_engineering.py` also appends boolean outlier flag columns (`age_outlier`, `glucose_outlier`, `bmi_outlier`) using IQR 1.5× fences — these are carried into `stroke_data_clean.csv` but are not used by the dashboard.

## Dashboard (`dashboard.py` + `analysis_utils.py`)

Seven tabs, each with an in-app `ℹ️` help expander:

| Tab | Description |
|-----|-------------|
| 🏥 Cohort Overview | Summary metrics, data quality report, and IQR outlier flag table |
| 📋 Table 1 | Clinical manuscript-style Table 1 stratified by stroke outcome, with CSV download |
| 📊 Distributions | Categorical: stacked % bar (stroke vs. no-stroke share per category). Numeric: overlapping count histogram + box plot split by outcome |
| 🎯 Individual Stroke Risk | Stroke rate per value/bin with 95% AC confidence intervals and significance color coding |
| 🔗 Joint Stroke Risk | Conditional stroke probability for a user-defined patient profile |
| 🔥 Correlation Matrix | Pearson correlation heatmap (categoricals ordinally encoded) |
| 🧪 Hypothesis Testing | Z-test results and forest plots for all features vs. the overall stroke rate |

### Key implementation notes

- `analysis_utils.ENCODE_MAP` is the **single source of truth** for ordinal encodings; it is used for both the correlation matrix display and as the reference for what `feature_engineering.py` replicates manually during KNN imputation. Keep them in sync if categories change.
- `dashboard.py` caches data with `@st.cache_data`. After re-running the pipeline scripts, clear the Streamlit cache (menu → Clear cache, or restart the server) to pick up fresh data.
- `load_data()` adds a synthetic `stroke_label` column (`{0: "No Stroke", 1: "Stroke"}`) that is not in the CSV. Code that exposes raw rows to users must drop it explicitly (e.g. `filtered.drop(columns=["stroke_label"])`).
- `CI_MIN_N = 30` (minimum group size to render CI bars) and the `hover_ci()` helper are defined in `dashboard.py`, not `analysis_utils.py`. `hover_ci` is defined inside the tab2 block but referenced in the tab5 forest plot — keep them in the same file if refactoring.
- In the Distributions tab, categorical features use a stacked 100% bar chart (`px.bar` with `barmode="stack"`, percentages computed manually before passing to Plotly). Numeric features show an overlapping histogram plus a `px.box` below it. The "Split by stroke outcome" checkbox is only rendered for numeric features; for categoricals the split is always shown.

### Statistical methods (`analysis_utils.py`)

- **Agresti-Coull CI** — confidence intervals for each group's stroke proportion; only rendered on charts when group N ≥ 30
- **Proportion z-test** — two-sided test against p₀ (overall stroke rate); significance stars: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns`
- Bar/forest plot color coding: 🔴 significantly higher risk, 🔵 significantly lower, ⬜ not significant
- **Table 1** (`table1_stats(df)`) — returns one row per feature plus indented sub-rows for categorical levels. Continuous: Mean (SD), two-sided Mann-Whitney U (`stats.mannwhitneyu`). Categorical: n (%), chi-square (`stats.chi2_contingency`) on the full contingency table; p-value on the parent row, blank on sub-rows. `_fmt_pvalue()` formats to `< 0.001 *`, `0.XXX *`, or `0.XXX`. Dashboard wraps this in `load_table1()` with `@st.cache_data`.
- **Phase 2 group comparison** — three functions comparing stroke=1 vs stroke=0 directly (not against population rate):
  - `mann_whitney_summary(df, feature)` — continuous features. Returns median + IQR per group, Mann-Whitney U + two-sided p-value, Cohen's d (pooled SD, positive = higher in stroke group), and magnitude label (`negligible` |d|<0.2, `small` 0.2–0.5, `medium` 0.5–0.8, `large` >0.8).
  - `chi_square_summary(df, feature)` — categorical features. Returns per-level n and stroke rate (%), chi-square + p-value, Cramér's V. For `BINARY_FEATURES` (`hypertension`, `heart_disease`) also adds odds ratio + 95% CI via `stats.contingency.odds_ratio` (scipy ≥ 1.7), oriented as feature=1 → stroke=1.
  - `phase2_summary(df)` — runs both over all features; returns a dict keyed by feature name.
- `BINARY_FEATURES = ["hypertension", "heart_disease"]` is a module-level constant used to gate odds ratio computation.

## `hypothesis_testing.py`

Standalone script that runs Phase 2 analysis and writes `data/phase2_hypothesis_results.csv`. Imports `phase2_summary` from `analysis_utils`; no test logic lives in this file.

- `build_results(summary)` — flattens the nested `phase2_summary` dict into one row per feature with columns: Feature, Test, No Stroke, Stroke, Statistic, p-value, Effect Size, Odds Ratio 95% CI, Sig. An internal `_p_raw` float column is used for filtering and dropped before CSV export.
- Categorical top-level "No Stroke" / "Stroke" summary columns show group N (`n=X,XXX`); per-level detail is in the `levels` sub-dict from `chi_square_summary` and is not repeated in the flat table.
- `print_findings(summary, results)` — plain-English summary of significant features. Numeric: reports direction (higher/lower median) and Cohen's d magnitude. Binary: reports OR direction. Multi-level categorical: reports Cramér's V only (no single direction).
- IQR ranges and OR CIs use plain hyphens (`-`) not en-dashes to avoid encoding issues on Windows terminals.

### Table 1 rendering notes

- Variable names are kept as a plain column (not the DataFrame index) to avoid non-unique index errors from identically-named sub-rows across features (e.g. `    0` appears under both `hypertension` and `heart_disease`). Use `hide_index=True` in `st.dataframe`.
- Significant p-values are bolded red via `display.style.map()` (not `applymap`, which is removed in pandas 2.1+). The styler checks for a trailing `*` in the string.
- The CSV download encodes the unstyled display DataFrame directly (`display.to_csv(index=False).encode()`).

### Numeric binning

Numeric features use width-based bins (aligned to multiples of the bin width). Defaults and available options:

| Feature | Default width | Options |
|---------|--------------|---------|
| `age` | 10 years | 1, 2, 3, 4, 5, 10, 15, 20 |
| `avg_glucose_level` | 50 mg/dL | 1, 5, 10, 15, 20, 25, 50, 75, 100 |
| `bmi` | 5 points | 1, 2, 3, 4, 5, 10, 15, 20 |

In the Joint Stroke Risk tab, continuous variables use min/max number inputs (defaulting to data bounds) instead of sliders.
