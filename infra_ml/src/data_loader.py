"""
data_loader.py
--------------
STEP 1 - Load the raw data.

Reading a 200,000-row Excel file is slow (about a minute), so the first
time we read it we also save a CSV copy. Every later run reads the CSV,
which takes only a few seconds.
"""
import pandas as pd

from src.config import EXCEL_FILE, CACHE_FILE


def load_raw_data():
    """Return the raw project data as a pandas DataFrame."""
    if CACHE_FILE.exists():
        # keep_default_na=False + na_values=[""] -> only truly empty cells
        # become missing. Without this pandas would turn the real category
        # "None" (no monitoring) into a missing value!
        return pd.read_csv(CACHE_FILE, keep_default_na=False, na_values=[""])

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {EXCEL_FILE}\n"
            "Put NIRIKSHAK_Projects_200000.xlsx inside the 'data' folder.")

    print("Reading Excel file (first run only, about a minute)...")
    df = pd.read_excel(EXCEL_FILE, sheet_name="Projects",
                       keep_default_na=False, na_values=[""])
    df.to_csv(CACHE_FILE, index=False)
    print(f"Saved fast CSV copy -> {CACHE_FILE.name}")
    return df


def describe_dataset(df):
    """Basic facts about the dataset (used by the UI and the console)."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": df.select_dtypes("number").shape[1],
        "text_columns": df.select_dtypes(exclude="number").shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_origin": ", ".join(df["data_origin"].unique()) if "data_origin" in df else "unknown",
    }


if __name__ == "__main__":
    data = load_raw_data()
    print(data.head())
    print(describe_dataset(data))
