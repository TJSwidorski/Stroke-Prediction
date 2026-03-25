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
pip install kagglehub[pandas-datasets] python-dotenv scikit-learn pandas numpy scipy streamlit plotly optuna joblib tensorflow shap
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
python train_logistic.py      # Logistic regression baseline → data/lr_model.pkl + data/lr_results.json
python train_mlp.py           # Four MLP configurations → data/mlp_*
python shap_analysis.py       # SHAP values + feature importance comparison → data/shap_* + data/feature_importance_comparison.csv
streamlit run dashboard.py        # Phase 1/2: descriptive analysis + hypothesis testing
streamlit run model_dashboard.py  # Phase 3: model results (separate app)
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
| bmi | Significantly higher in stroke patients (median 29.5 vs. 28.1 kg/m², p < 0.001) | Cohen's d = 0.185 (negligible) |
| work_type | Distribution differs by stroke outcome (p < 0.001) | Cramér's V = 0.098 (negligible) |
| smoking_status | Distribution differs by stroke outcome (p < 0.001) | Cramér's V = 0.055 (negligible) |

2 features did not reach statistical significance (p ≥ 0.05): `gender` and `Residence_type`.

All associations are observational; no causal claims are made. See `data/phase2_interpretation.txt` for full manuscript-style interpretation.

## Phase 3 — Predictive Modeling

Models are trained on the 8 statistically significant features identified in Phase 2 (`age`, `avg_glucose_level`, `heart_disease`, `hypertension`, `ever_married`, `bmi`, `work_type`, `smoking_status`). Feature selection is driven by `data/phase2_hypothesis_results.csv`, so `hypothesis_testing.py` must run first.

Preprocessing (shared via `preprocessing.py`):
- Binary string features (`ever_married`, `Residence_type`) mapped to 0/1
- Multi-level categoricals (`work_type`, `smoking_status`) one-hot encoded (no reference drop)
- Numeric features (`age`, `bmi`, `avg_glucose_level`) standardized; fitted scaler saved to `data/scaler.pkl`
- Stratified 80/20 train/test split; balanced class weights applied during training to address ~5% stroke prevalence

**Logistic regression baseline** (`train_logistic.py`) — ElasticNet regularization (`penalty="elasticnet"`, solver `"saga"`), hyperparameters tuned via 100-trial Optuna Bayesian search maximizing 5-fold CV AUC-ROC.

**MLP configurations** (`train_mlp.py`) — four fixed architectures trained with early stopping (patience 15, monitor `val_auc`) and learning rate reduction on plateau:

| Config | Architecture | Regularization |
|--------|-------------|----------------|
| Shallow Wide | [128] | None |
| Medium Dropout | [64, 32] | Dropout 0.3 |
| Deep Regularized | [128, 64, 32] | Dropout 0.3 + L2 0.001 |
| Attention Weighted | [64, 32] | Dropout 0.2 + input attention |

The Attention Weighted config prepends a `softmax` Dense layer that learns per-feature importance weights, multiplied element-wise with the inputs. These weights are averaged across training samples and saved to `data/mlp_attention_weights.json`.

All models are evaluated at two decision thresholds: default 0.5 and an optimal threshold targeting ≥ 85% sensitivity (highest threshold on the training ROC curve still meeting that target, maximizing specificity), reflecting the clinical priority of minimizing missed strokes.

**SHAP feature importance** (`shap_analysis.py`) — runs after both model scripts and produces:
- Per-model SHAP values and beeswarm/bar plots for the LR baseline and best MLP (by AUC-ROC)
- `data/feature_importance_comparison.csv` — normalized [0, 1] ranking across LR coefficients, LR SHAP, MLP SHAP, and MLP attention weights, enabling direct comparison of which features drive stroke risk across all methods

## Interactive Dashboards

Two separate Streamlit apps cover different phases of the project.

### Phase 1/2 dashboard (`dashboard.py`)

The Streamlit dashboard provides eight analysis tabs:

- **Cohort Overview** — Summary metrics, data quality report, and IQR outlier flag counts
- **Table 1** — Clinical manuscript-style Table 1 stratified by stroke outcome; continuous variables as Mean (SD) with Mann-Whitney U p-values, categorical variables as n (%) with chi-square p-values; CSV download included
- **Distributions** — Categorical features as stacked percentage bars (stroke share per category); numeric features as overlapping count histogram plus a box plot split by outcome
- **Individual Stroke Risk** — Stroke rate per feature value or bin, with 95% Agresti-Coull confidence intervals and hypothesis test color coding (red = significantly higher risk, blue = significantly lower, gray = not significant)
- **Joint Stroke Risk** — Conditional stroke probability for a custom patient profile, shown on a gauge vs. the population baseline
- **Correlation Matrix** — Pearson correlation heatmap across all features
- **Hypothesis Testing** — Two-sided z-tests of each feature value's stroke rate against the overall population rate (p₀), with z-statistic bar chart and per-feature forest plots
- **Phase 2 — Group Comparison** — Mann-Whitney U and chi-square tests comparing stroke vs. no-stroke groups directly, with odds ratios, effect sizes (Cohen's d / Cramér's V), a styled results table, per-feature detail charts, and an OR forest plot covering both binary and dummy-coded categorical variables

Each tab includes an in-app help section explaining how to interpret the results.

### Phase 3 dashboard (`model_dashboard.py`)

Reports all modeling results transparently, showing what happened under the hood:

- **Model Overview** — all five models (LR + 4 MLPs) side by side at both default and optimal thresholds; best-model cards; plain-English threshold explanation
- **Training Curves** — all four MLP configs on the same axes (AUC-ROC and loss); per-config detailed view with early stopping epoch marked
- **Logistic Regression Detail** — Optuna optimization history, hyperparameter search space scatter, coefficient and odds-ratio plots, confusion matrices at both thresholds
- **Neural Network Detail** — per-config architecture table, training subplots, confusion matrices, attention weight chart (Config D)
- **SHAP Interpretability** — beeswarm and bar plots for LR and best MLP; feature importance comparison table with gradient styling; per-feature cross-method rank cards
- **Methodology** — written rationale for every modeling decision; data flow diagram; limitations

A sidebar shows file status (✅/❌) for every expected output file with the command needed to generate it.
