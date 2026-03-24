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
3. **`dashboard.py`** — Streamlit interactive analysis dashboard (reads `data/stroke_data_clean.csv`)

```bash
python retrieve_data.py
python feature_engineering.py
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

Six tabs, each with an in-app `ℹ️` help expander:

| Tab | Description |
|-----|-------------|
| 🏥 Cohort Overview | Summary metrics, data quality report, and IQR outlier flag table |
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

### Numeric binning

Numeric features use width-based bins (aligned to multiples of the bin width). Defaults and available options:

| Feature | Default width | Options |
|---------|--------------|---------|
| `age` | 10 years | 1, 2, 3, 4, 5, 10, 15, 20 |
| `avg_glucose_level` | 50 mg/dL | 1, 5, 10, 15, 20, 25, 50, 75, 100 |
| `bmi` | 5 points | 1, 2, 3, 4, 5, 10, 15, 20 |

In the Joint Stroke Risk tab, continuous variables use min/max number inputs (defaulting to data bounds) instead of sliders.
