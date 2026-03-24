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
3. **`hypothesis_testing.py`** — Phase 2 group comparison analysis, prints results to console, saves to `data/phase2_hypothesis_results.csv`, and writes a manuscript-style interpretation to `data/phase2_interpretation.txt`
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

Eight tabs, each with an in-app `ℹ️` help expander:

| Tab | Description |
|-----|-------------|
| 🏥 Cohort Overview | Summary metrics, data quality report, and IQR outlier flag table |
| 📋 Table 1 | Clinical manuscript-style Table 1 stratified by stroke outcome, with CSV download |
| 📊 Distributions | Categorical: stacked % bar (stroke vs. no-stroke share per category). Numeric: overlapping count histogram + box plot split by outcome |
| 🎯 Individual Stroke Risk | Stroke rate per value/bin with 95% AC confidence intervals and significance color coding |
| 🔗 Joint Stroke Risk | Conditional stroke probability for a user-defined patient profile |
| 🔥 Correlation Matrix | Pearson correlation heatmap (categoricals ordinally encoded) |
| 🧪 Hypothesis Testing | Z-test results and forest plots for all features vs. the overall stroke rate |
| 📊 Phase 2 — Group Comparison | Metric cards, color-coded results table, and per-feature detail charts from `phase2_hypothesis_results.csv` |

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
  - `chi_square_summary(df, feature)` — categorical features. Returns per-level n and stroke rate (%), per-level odds ratios (dummy coded vs reference category), chi-square + p-value, Cramér's V, and `reference_category`. For `BINARY_FEATURES` also adds a top-level OR via `stats.contingency.odds_ratio` (scipy ≥ 1.7) using `contingency.values` directly (works for both integer and string-indexed tables). Reference category defaults to first sorted value; overridden by `REFERENCE_CATEGORIES`.
  - `phase2_summary(df)` — runs both over all features; returns a dict keyed by feature name.
- `BINARY_FEATURES = ["hypertension", "heart_disease", "ever_married", "Residence_type"]` — gates top-level OR computation. `ever_married` ("No"/"Yes") and `Residence_type` ("Rural"/"Urban") are string-valued but treated as binary.
- `REFERENCE_CATEGORIES = {"work_type": "Private", "smoking_status": "never smoked"}` — overrides the default first-sorted reference for dummy-coded per-level ORs. Add entries here to change any feature's reference without touching the function logic.

## `hypothesis_testing.py`

Standalone script that runs Phase 2 analysis and writes `data/phase2_hypothesis_results.csv`. Imports `phase2_summary` from `analysis_utils`; no test logic lives in this file.

- `build_results(summary)` — flattens the nested `phase2_summary` dict into one row per feature with columns: Feature, Test, No Stroke, Stroke, Statistic, p-value, Effect Size, Odds Ratio 95% CI, Sig. An internal `_p_raw` float column is used for filtering and dropped before CSV export.
- Categorical top-level "No Stroke" / "Stroke" summary columns show group N (`n=X,XXX`); per-level detail is in the `levels` sub-dict from `chi_square_summary` and is not repeated in the flat table.
- `print_findings(summary, results)` — plain-English summary of significant features. Numeric: reports direction (higher/lower median) and Cohen's d magnitude. Binary (all 4): reports OR direction. Multi-level categorical: reports Cramér's V and iterates per-level ORs vs the reference category.
- `write_interpretation(results_df, summary)` — writes `data/phase2_interpretation.txt` structured as: (1) header with run timestamp, (2) Primary findings — top 3 features ranked by effect size (`|d|` for numeric, Cramér's V for categorical), each with 2-3 manuscript-style sentences including a plain-English clinical implication, (3) Secondary findings — one sentence per remaining significant feature, (4) Non-significant features — brief list, (5) Limitations paragraph covering cross-sectional design, class imbalance, no multiple-comparison correction, and ordinal encoding caveat. Effect size ranking uses `_effect_sort_key()` which parses `"d=X.XXX (...)"` and `"V=X.XXX"` strings from the results DataFrame. Numeric sentences include per-feature units (years / mg/dL / kg/m²). File written with `encoding="utf-8"`.
- IQR ranges and OR CIs use plain hyphens (`-`) not en-dashes to avoid encoding issues on Windows terminals.

### Phase 2 tab implementation notes

- Reads `data/phase2_hypothesis_results.csv` at render time (not cached). If the file is missing, shows a warning and skips the rest of the tab content via an `if p2 is not None:` guard — do not use `st.stop()` inside a tab block, as it halts the entire app script and prevents other tabs from rendering.
- `_parse_p(s)` converts formatted p-value strings (`"< 0.001"` → `0.0005`, else `float(s)`) to enable numeric comparisons for the metric cards.
- OR > 2.0 metric extracts the point estimate by splitting the `"X.XX (Y.YY-Z.ZZ)"` string on whitespace and parsing the first token.
- Significance column is colored via `style.map(_color_sig)`: `***` red, `**` orange, `*` dark yellow, `ns` gray.
- Feature detail: numeric features use `px.box` on the live `df`; categorical features call `feature_stroke_stats()` for the bar chart and `load_phase2_detail()` for a per-level OR table below it. `load_phase2_detail()` is a `@st.cache_data` wrapper around `phase2_summary(load_data())`.
- Interpretations expander (below the download button): iterates `CATEGORICAL` features. Binary features get one sentence using the top-level OR. Multi-level significant features get one bullet per category level (reference noted, non-reference levels get OR sentences via `_or_sentence()`). Non-significant multi-level features get a single no-association note. The `_or_sentence()` helper appends `(not statistically significant)` when `p_raw ≥ 0.05`.
- **OR Forest Plot** (below the feature detail expander): combines two OR sources into a single `fp_frame` DataFrame.
  - Binary features: top-level OR and CI parsed from the `phase2_hypothesis_results.csv` "Odds Ratio 95% CI" column via `_parse_or_ci(s)` (splits on whitespace, strips parens from CI token, splits on `-`).
  - Non-binary categoricals: per-level dummy-coded ORs from `load_phase2_detail()` — reference levels shown as `"diamond"` markers at OR=1.0 with zeroed error bars and label `"(reference)"`; non-reference levels shown as `"circle"` markers.
  - Sort order: `group_ord` descending (binary features sorted last → appear at top), `within_ord` ascending within each feature group. `group_ord` uses `CATEGORICAL.index(feat)` to preserve list order for categoricals.
  - Per-level color: `lo > 1.0` → red, `hi < 1.0` → blue, else gray. Significance for per-level ORs is determined by CI excluding 1.0, not the overall chi-square p-value.
  - Levels where OR is NaN (e.g. zero-cell cells like `Never_worked` in work_type) are skipped.
  - Log-scale x-axis with OR=1.0 reference line. `height=max(350, len(fp_frame) * 70)`, `margin=dict(r=230)`.
  - Caption dynamically lists reference categories for each multi-level feature by reading `p2_det_fp[f]["reference_category"]` from `load_phase2_detail()`.

### Table 1 rendering notes

- Variable names are kept as a plain column (not the DataFrame index) to avoid non-unique index errors from identically-named sub-rows across features (e.g. `    0` appears under both `hypertension` and `heart_disease`). Use `hide_index=True` in `st.dataframe`.
- Significant p-values are bolded red via `display.style.map()` (not `applymap`, which is removed in pandas 2.1+). The styler checks for a trailing `*` in the string.
- The CSV download encodes the unstyled display DataFrame directly (`display.to_csv(index=False).encode()`).
- An `ℹ️ How to read this table` expander sits directly above the dataframe. It explains the Mean (SD) + Mann-Whitney pairing rationale (right-skewed distributions violate t-test normality; Mean/SD retained for comparability per clinical convention), chi-square for categoricals, and the `*` significance marker.
- The `st.caption` below the table explicitly notes that Mean (SD) is for descriptive comparability and that test selection follows distributional properties — addressing the ambiguity of showing a parametric descriptive alongside a non-parametric p-value.

### Numeric binning

Numeric features use width-based bins (aligned to multiples of the bin width). Defaults and available options:

| Feature | Default width | Options |
|---------|--------------|---------|
| `age` | 10 years | 1, 2, 3, 4, 5, 10, 15, 20 |
| `avg_glucose_level` | 50 mg/dL | 1, 5, 10, 15, 20, 25, 50, 75, 100 |
| `bmi` | 5 points | 1, 2, 3, 4, 5, 10, 15, 20 |

In the Joint Stroke Risk tab, continuous variables use min/max number inputs (defaulting to data bounds) instead of sliders.
