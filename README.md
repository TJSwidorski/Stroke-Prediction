# Stroke-Prediction

This project applies a clinical medical research workflow to the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) to identify which combination of cardiovascular and lifestyle risk factors can be used to predict strokes, and determine if a model can be used to identify high-risk patients prior to the event.

## Data source

Soriano, F. (2021). *Stroke Prediction Dataset* [Data set]. Kaggle.
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Original data sourced from a McKinsey & Company healthcare analytics competition
via Analytics Vidhya. The dataset contains 5,110 de-identified patient records
with 11 clinical features.

## Setup

```bash
pip install kagglehub[pandas-datasets] python-dotenv scikit-learn pandas numpy scipy streamlit plotly
```

Create a `.env` file in the project root with your Kaggle credentials:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Running the Pipeline

```bash
python retrieve_data.py       # Download dataset → data/stroke_data.csv
python feature_engineering.py # Impute missing values → data/stroke_data_clean.csv + data/data_quality_report.txt
python hypothesis_testing.py  # Group comparison analysis → data/phase2_hypothesis_results.csv + data/phase2_interpretation.txt
streamlit run dashboard.py    # Launch interactive dashboard
```

## Data Preprocessing

Missing and unknown values are handled in `feature_engineering.py` using K-Nearest Neighbors (k=5):
- **BMI** — NaN values imputed based on similar patients
- **Smoking status** — `"Unknown"` entries resolved to the most common status among nearest neighbors

## Key Findings — Phase 2

Group comparison tests (Mann-Whitney U and chi-square) were run comparing stroke vs. no-stroke patients across all 10 clinical features.

| Feature | Result | Effect Size |
|---------|--------|-------------|
| age | Significantly higher in stroke patients (median 71.0 vs. 43.0 years, p < 0.001) | Cohen's d = 1.175 (large) |
| avg_glucose_level | Significantly higher in stroke patients (median 105.2 vs. 91.5 mg/dL, p < 0.001) | Cohen's d = 0.618 (medium) |
| heart_disease | Higher odds of stroke (OR = 4.70, 95% CI 3.26–6.69, p < 0.001) | Cramér's V = 0.133 (small) |
| hypertension | Higher odds of stroke (OR = 3.70, 95% CI 2.70–5.02, p < 0.001) | Cramér's V = 0.126 (small) |
| ever_married | Higher odds of stroke (OR = 4.18, 95% CI 2.82–6.42, p < 0.001) | Cramér's V = 0.107 (small) |
| work_type | Distribution differs by stroke outcome (p < 0.001) | Cramér's V = 0.098 (negligible) |
| bmi | Significantly higher in stroke patients (median 29.5 vs. 28.1 kg/m², p < 0.001) | Cohen's d = 0.185 (negligible) |
| smoking_status | Distribution differs by stroke outcome (p < 0.001) | Cramér's V = 0.055 (negligible) |

2 features did not reach statistical significance (p ≥ 0.05): `gender` and `Residence_type`.

All associations are observational; no causal claims are made. See `data/phase2_interpretation.txt` for full manuscript-style interpretation.

## Interactive Dashboard

The Streamlit dashboard (`dashboard.py`) provides eight analysis tabs:

- **Cohort Overview** — Summary metrics, data quality report, and IQR outlier flag counts
- **Table 1** — Clinical manuscript-style Table 1 stratified by stroke outcome; continuous variables as Mean (SD) with Mann-Whitney U p-values, categorical variables as n (%) with chi-square p-values; CSV download included
- **Distributions** — Categorical features as stacked percentage bars (stroke share per category); numeric features as overlapping count histogram plus a box plot split by outcome
- **Individual Stroke Risk** — Stroke rate per feature value or bin, with 95% Agresti-Coull confidence intervals and hypothesis test color coding (red = significantly higher risk, blue = significantly lower, gray = not significant)
- **Joint Stroke Risk** — Conditional stroke probability for a custom patient profile, shown on a gauge vs. the population baseline
- **Correlation Matrix** — Pearson correlation heatmap across all features
- **Hypothesis Testing** — Two-sided z-tests of each feature value's stroke rate against the overall population rate (p₀), with z-statistic bar chart and per-feature forest plots
- **Phase 2 — Group Comparison** — Mann-Whitney U and chi-square tests comparing stroke vs. no-stroke groups directly, with odds ratios, effect sizes (Cohen's d / Cramér's V), a styled results table, per-feature detail charts, and an OR forest plot covering both binary and dummy-coded categorical variables

Each tab includes an in-app help section explaining how to interpret the results.
