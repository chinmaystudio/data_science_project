"""
data_cleaning.py
----------------
STEP 2 - Clean the raw data.

What is done here (in order):
  1. Remove exact duplicate rows.
  2. Fix data types (text dates -> real dates).
  3. Check for impossible values (negative cost, land % above 100 ...).
  4. Report and treat missing values:
       - numeric predictors -> filled with the MEDIAN (robust to outliers)
       - completion date is left empty on purpose: an empty date simply
         means the project is not finished yet (meaningful missing value)
  5. Create the classification target 'major_delay'.
The cleaning report (what was found / changed) is returned so the UI can
show it.
"""
import pandas as pd

from src.config import (CLEAN_FILE, NUMERIC_FEATURES, MAJOR_DELAY_MONTHS,
                        CLASS_TARGET)


def missing_value_table(df):
    """Count and percentage of missing values for every column."""
    table = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    return table[table["missing_count"] > 0].sort_values("missing_count",
                                                         ascending=False)


def invalid_value_checks(df):
    """Count rows that break simple common-sense rules."""
    checks = {
        "Sanctioned cost <= 0": (df["sanctioned_cost_inr_cr"] <= 0).sum(),
        "Planned duration <= 0": (df["planned_duration_months"] <= 0).sum(),
        "Land acquired % outside 0-100":
            ((df["land_acquired_pct_at_award"] < 0) |
             (df["land_acquired_pct_at_award"] > 100)).sum(),
        "Negative delay": (df["delay_months"] < 0).sum(),
        "Bid ratio <= 0": (df["bid_to_estimate_ratio"] <= 0).sum(),
        "Award date after 30-09-2026 (status date)":
            (pd.to_datetime(df["award_date"]) > "2026-09-30").sum(),
        "Not-started project (Tendered/Approved) with delay > 0":
            (df["status"].isin(["TENDERED", "APPROVED"]) &
             (df["delay_months"] > 0)).sum(),
    }
    return pd.DataFrame({"check": list(checks.keys()),
                         "rows_found": [int(v) for v in checks.values()]})


def clean_data(raw_df, save=True):
    """Return (clean DataFrame, cleaning report dict)."""
    df = raw_df.copy()             # never change the original data
    report = {}

    # 1. duplicates
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. data types
    for col in ["award_date", "planned_completion_date",
                "actual_or_expected_completion_date", "actual_completion_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()                 # remove stray spaces

    # 3. invalid values (only reported - none are impossible values;
    #    the two logical issues come from the synthetic data and are kept)
    report["invalid_checks"] = invalid_value_checks(df)

    # 4. missing values
    report["missing_before"] = missing_value_table(df)
    filled = {}
    for col in NUMERIC_FEATURES:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            filled[col] = {"filled_rows": n_missing,
                           "median_used": round(float(median_value), 2)}
    report["median_filled"] = pd.DataFrame(filled).T
    report["missing_after"] = missing_value_table(df)

    # 5. classification target
    df[CLASS_TARGET] = (df["delay_months"] > MAJOR_DELAY_MONTHS).astype(int)
    report["major_delay_rate_pct"] = round(df[CLASS_TARGET].mean() * 100, 2)

    report["rows_after_cleaning"] = len(df)
    if save:
        df.to_csv(CLEAN_FILE, index=False)
    return df, report


def load_clean_data():
    """Read the cleaned file (created by train.py / clean_data)."""
    return pd.read_csv(CLEAN_FILE, keep_default_na=False, na_values=[""],
                       parse_dates=["award_date"])


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    clean, rep = clean_data(load_raw_data())
    print(rep["invalid_checks"])
    print(rep["missing_before"])
    print("Major delay rate:", rep["major_delay_rate_pct"], "%")
