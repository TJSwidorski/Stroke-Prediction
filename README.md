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
streamlit run dashboard.py    # Launch interactive dashboard
```

## Data Preprocessing

Missing and unknown values are handled in `feature_engineering.py` using K-Nearest Neighbors (k=5):
- **BMI** — NaN values imputed based on similar patients
- **Smoking status** — `"Unknown"` entries resolved to the most common status among nearest neighbors

## Interactive Dashboard

The Streamlit dashboard (`dashboard.py`) provides seven analysis tabs:

- **Cohort Overview** — Summary metrics, data quality report, and IQR outlier flag counts
- **Table 1** — Clinical manuscript-style Table 1 stratified by stroke outcome; continuous variables as Mean (SD) with Mann-Whitney U p-values, categorical variables as n (%) with chi-square p-values; CSV download included
- **Distributions** — Categorical features as stacked percentage bars (stroke share per category); numeric features as overlapping count histogram plus a box plot split by outcome
- **Individual Stroke Risk** — Stroke rate per feature value or bin, with 95% Agresti-Coull confidence intervals and hypothesis test color coding (red = significantly higher risk, blue = significantly lower, gray = not significant)
- **Joint Stroke Risk** — Conditional stroke probability for a custom patient profile, shown on a gauge vs. the population baseline
- **Correlation Matrix** — Pearson correlation heatmap across all features
- **Hypothesis Testing** — Two-sided z-tests of each feature value's stroke rate against the overall population rate (p₀), with z-statistic bar chart and per-feature forest plots

Each tab includes an in-app help section explaining how to interpret the results.
