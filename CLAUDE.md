# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a clinical medical research project that applies machine learning to the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`fedesoriano/stroke-prediction-dataset` on Kaggle). The goal is to identify which cardiovascular and lifestyle risk factors predict strokes and build a model to flag high-risk patients.

## Setup

Install dependencies:
```bash
pip install kagglehub[pandas-datasets] python-dotenv scikit-learn pandas numpy streamlit plotly
```

Kaggle credentials go in a `.env` file (excluded from git):
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

## Pipeline

Run scripts in order:

1. **`retrieve_data.py`** — downloads dataset from Kaggle, saves to `data/stroke_data.csv`
2. **`feature_engineering.py`** — KNN imputes missing BMI values and Unknown smoking status, saves to `data/stroke_data_clean.csv`
3. **`dashboard.py`** — Streamlit interactive analysis dashboard (reads `data/stroke_data_clean.csv`)

```bash
python retrieve_data.py
python feature_engineering.py
streamlit run dashboard.py
```

`data/` is gitignored — raw and cleaned CSVs are not committed.

## Dashboard

`dashboard.py` provides four tabs:
- **Distributions** — histogram per feature, optionally split by stroke outcome
- **Individual Stroke Risk** — stroke rate per feature value/bin with overall baseline reference line
- **Joint Stroke Risk** — multiselect features, set values/ranges, get conditional stroke probability with gauge chart
- **Correlation Matrix** — Pearson correlation heatmap (categoricals ordinally encoded); sortable stroke-correlation table

`analysis_utils.py` contains the computation helpers (binning, filtering, encoding) used by the dashboard.

## Data

Key columns: `age`, `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke` (target).

Known data quality issues (handled in `feature_engineering.py`):
- `bmi`: contains NaN values → filled via KNN (k=5)
- `smoking_status`: contains `"Unknown"` entries → filled via KNN (k=5) using encoded features
