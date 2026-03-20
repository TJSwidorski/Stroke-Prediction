# Stroke-Prediction

This project applies a clinical medical research workflow to the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) to identify which combination of cardiovascular and lifestyle risk factors can be used to predict strokes, and determine if a model can be used to identify high-risk patients prior to the event.

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
python feature_engineering.py # Impute missing values → data/stroke_data_clean.csv
streamlit run dashboard.py    # Launch interactive dashboard
```

## Data Preprocessing

Missing and unknown values are handled in `feature_engineering.py` using K-Nearest Neighbors (k=5):
- **BMI** — NaN values imputed based on similar patients
- **Smoking status** — `"Unknown"` entries resolved to the most common status among nearest neighbors

## Interactive Dashboard

The Streamlit dashboard (`dashboard.py`) provides five analysis tabs:

- **Distributions** — Patient counts per feature, with optional stroke outcome split
- **Individual Stroke Risk** — Stroke rate per feature value or bin, with 95% Agresti-Coull confidence intervals and hypothesis test color coding (red = significantly higher risk, blue = significantly lower, gray = not significant)
- **Joint Stroke Risk** — Conditional stroke probability for a custom patient profile, shown on a gauge vs. the population baseline
- **Correlation Matrix** — Pearson correlation heatmap across all features
- **Hypothesis Testing** — Two-sided z-tests of each feature value's stroke rate against the overall population rate (p₀), with z-statistic bar chart and per-feature forest plots

Each tab includes an in-app help section explaining how to interpret the results.
