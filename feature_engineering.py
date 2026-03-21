import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import KNNImputer

df = pd.read_csv("data/stroke_data.csv")

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
original_shape = df.shape

# --- Outlier detection (IQR method, 1.5x fence) ---
OUTLIER_VARS = {
    "age":               "age_outlier",
    "avg_glucose_level": "glucose_outlier",
    "bmi":               "bmi_outlier",
}

total_flagged_rows = set()
outlier_counts = {}

for col, flag_col in OUTLIER_VARS.items():
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    df[flag_col] = outlier_mask
    flagged = df.loc[outlier_mask, col]
    total_flagged_rows.update(df.index[outlier_mask].tolist())
    outlier_counts[col] = int(outlier_mask.sum())

    print(f"[{col}]")
    print(f"  IQR bounds : {lower:.2f} – {upper:.2f}  (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})")
    print(f"  Flagged rows: {outlier_mask.sum()}")
    if outlier_mask.sum() > 0:
        print(f"  Flagged range: {flagged.min():.2f} – {flagged.max():.2f}")
    print()

print(f"{len(total_flagged_rows)} total rows flagged across all variables.")
print()

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
bmi_nan_count = int(df["bmi"].isna().sum())
bmi_pre_mean = df["bmi"].mean()
bmi_pre_std = df["bmi"].std()

df_enc["smoking_status"] = df_enc["smoking_status"].map(smoking_map)  # Unknown -> NaN temporarily

imputer_bmi = KNNImputer(n_neighbors=5)
df_enc[features] = imputer_bmi.fit_transform(df_enc[features])

df["bmi"] = df_enc["bmi"]
bmi_post_mean = df["bmi"].mean()
bmi_post_std = df["bmi"].std()
print(f"BMI NaNs remaining: {df['bmi'].isna().sum()}")

# --- Step 2: KNN impute smoking_status ---
# Re-encode (bmi now filled, smoking_status is numeric with NaN for Unknown)
smoking_unknown_mask = df["smoking_status"] == "Unknown"
smoking_unknown_count = int(smoking_unknown_mask.sum())

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
smoking_resolved_breakdown = df.loc[smoking_unknown_mask, "smoking_status"].value_counts().to_dict()
print(f"Smoking 'Unknown' remaining: {(df['smoking_status'] == 'Unknown').sum()}")

# --- Confirm no missing values remain ---
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

df.to_csv("data/stroke_data_clean.csv", index=False)
print("\nSaved cleaned data to data/stroke_data_clean.csv")

# --- Write data quality report ---
final_missing = int(df.isnull().sum().sum())
report_lines = [
    "DATA QUALITY REPORT",
    "===================",
    "",
    f"Run timestamp : {run_timestamp}",
    "",
    "ORIGINAL DATASET",
    "----------------",
    f"  Rows    : {original_shape[0]}",
    f"  Columns : {original_shape[1]}",
    "",
    "BMI IMPUTATION (KNN, k=5)",
    "-------------------------",
    f"  NaN values imputed : {bmi_nan_count}",
    f"  Pre-imputation  mean : {bmi_pre_mean:.4f}  std : {bmi_pre_std:.4f}",
    f"  Post-imputation mean : {bmi_post_mean:.4f}  std : {bmi_post_std:.4f}",
    "",
    "SMOKING STATUS IMPUTATION (KNN, k=5)",
    "-------------------------------------",
    f"  'Unknown' values resolved : {smoking_unknown_count}",
    "  Resolved to:",
]
for category in ["never smoked", "formerly smoked", "smokes"]:
    count = smoking_resolved_breakdown.get(category, 0)
    report_lines.append(f"    {category:<20} : {count}")
report_lines += [
    "",
    "OUTLIER FLAGS (IQR 1.5x fence)",
    "------------------------------",
]
for col in OUTLIER_VARS:
    report_lines.append(f"  {col:<22} : {outlier_counts[col]} rows flagged")
report_lines += [
    f"  {'(total unique rows)':<22} : {len(total_flagged_rows)} rows flagged across all variables",
    "",
    "FINAL DATASET",
    "-------------",
    f"  Rows    : {df.shape[0]}",
    f"  Columns : {df.shape[1]}",
    f"  Missing values : {final_missing} ({'PASS' if final_missing == 0 else 'FAIL — missing values remain'})",
]

report_text = "\n".join(report_lines) + "\n"
with open("data/data_quality_report.txt", "w") as f:
    f.write(report_text)
print("Saved data quality report to data/data_quality_report.txt")