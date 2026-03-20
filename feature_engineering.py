import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv("data/stroke_data.csv")

# --- Encode categoricals for KNN distance calculations ---
gender_map = {"Male": 0, "Female": 1, "Other": 2}
married_map = {"No": 0, "Yes": 1}
work_map = {"children": 0, "Never_worked": 1, "Govt_job": 2, "Private": 3, "Self-employed": 4}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2}  # Unknown -> NaN

df_enc = df.copy()
df_enc["gender"] = df_enc["gender"].map(gender_map)
df_enc["ever_married"] = df_enc["ever_married"].map(married_map)
df_enc["work_type"] = df_enc["work_type"].map(work_map)
df_enc["Residence_type"] = df_enc["Residence_type"].map(residence_map)

# Drop id and stroke (target) before imputation
features = ["gender", "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]

# --- Step 1: KNN impute BMI ---
df_enc["smoking_status"] = df_enc["smoking_status"].map(smoking_map)  # Unknown -> NaN temporarily

imputer_bmi = KNNImputer(n_neighbors=5)
df_enc[features] = imputer_bmi.fit_transform(df_enc[features])

df["bmi"] = df_enc["bmi"]
print(f"BMI NaNs remaining: {df['bmi'].isna().sum()}")

# --- Step 2: KNN impute smoking_status ---
# Re-encode (bmi now filled, smoking_status is numeric with NaN for Unknown)
df_enc2 = df.copy()
df_enc2["gender"] = df_enc2["gender"].map(gender_map)
df_enc2["ever_married"] = df_enc2["ever_married"].map(married_map)
df_enc2["work_type"] = df_enc2["work_type"].map(work_map)
df_enc2["Residence_type"] = df_enc2["Residence_type"].map(residence_map)
df_enc2["smoking_status"] = df_enc2["smoking_status"].map(smoking_map)  # Unknown -> NaN

imputer_smoking = KNNImputer(n_neighbors=5)
df_enc2[features] = imputer_smoking.fit_transform(df_enc2[features])

# Round to nearest valid category (0, 1, 2) and decode
smoking_inv = {0: "never smoked", 1: "formerly smoked", 2: "smokes"}
df["smoking_status"] = df_enc2["smoking_status"].round().clip(0, 2).astype(int).map(smoking_inv)
print(f"Smoking 'Unknown' remaining: {(df['smoking_status'] == 'Unknown').sum()}")

# --- Confirm no missing values remain ---
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

df.to_csv("data/stroke_data_clean.csv", index=False)
print("\nSaved cleaned data to data/stroke_data_clean.csv")
# print(df.head())