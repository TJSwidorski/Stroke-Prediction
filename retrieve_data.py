# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] python-dotenv
import os
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter

load_dotenv()

os.makedirs("data", exist_ok=True)

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/stroke-prediction-dataset",
    "",
)

df.to_csv("data/stroke_data.csv", index=False)
print(f"Saved {len(df)} rows to data/stroke_data.csv")
print(df.head())
