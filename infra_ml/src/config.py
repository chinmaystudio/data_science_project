"""
config.py
---------
One place for all settings: file paths, column lists and the two
prediction targets. Every other file imports from here, so if a path or
column name changes you only edit this file.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EXCEL_FILE = DATA_DIR / "NIRIKSHAK_Projects_200000.xlsx"
CACHE_FILE = DATA_DIR / "projects_cache.csv"     # fast copy made on first load
CLEAN_FILE = DATA_DIR / "projects_clean.csv"     # output of data_cleaning.py

RANDOM_STATE = 42      # fixed seed -> same split / same results every run
TEST_SIZE = 0.20       # 80 % training, 20 % testing

# ---------------------------------------------------------------- targets
# Classification: will the project suffer a MAJOR delay (more than 18 months)?
MAJOR_DELAY_MONTHS = 18
CLASS_TARGET = "major_delay"

# Regression: how much will the cost go above the sanctioned cost (%)?
REG_TARGET = "cost_overrun_pct"

# ---------------------------------------------------------------- features
# Only information that is known when the project is awarded / planned.
# Columns that are calculated FROM the outcome (revised cost, delay months,
# completion dates, status ...) are NOT used -> this avoids "data leakage".
NUMERIC_FEATURES = [
    "sanctioned_cost_inr_cr",
    "planned_duration_months",
    "complexity_index",
    "num_bidders",
    "bid_to_estimate_ratio",
    "contractor_experience_yrs",
    "contractor_completed_projects",
    "land_required_ha",
    "land_acquired_pct_at_award",
    "utility_shifting_required",
    "design_change_count",
    "dpr_quality_score",
    "avg_annual_rainfall_mm",
    "political_change_during_exec",
    "litigation_flag",
    "covid_affected",
    "avg_payment_delay_days",
    "inflation_rate_pct",
    "material_price_index",
]

CATEGORICAL_FEATURES = [
    "sector",
    "area_type",
    "terrain",
    "contract_model",
    "funding_source",
    "monitoring_mechanism",
    "env_clearance",
    "forest_clearance",
]

# Columns that would leak the answer into the model (never used as inputs)
LEAKAGE_COLUMNS = [
    "delay_months", "time_overrun_pct", "is_delayed", "delay_category",
    "revised_cost_inr_cr", "cost_overrun_pct", "is_cost_overrun", "status",
    "actual_or_expected_completion_date", "actual_completion_date",
]

# Friendly names used in charts and in the UI
PRETTY = {
    "sanctioned_cost_inr_cr": "Sanctioned cost (INR Cr)",
    "planned_duration_months": "Planned duration (months)",
    "complexity_index": "Complexity index",
    "num_bidders": "Number of bidders",
    "bid_to_estimate_ratio": "Bid / estimate ratio",
    "contractor_experience_yrs": "Contractor experience (yrs)",
    "contractor_completed_projects": "Contractor completed projects",
    "land_required_ha": "Land required (ha)",
    "land_acquired_pct_at_award": "Land acquired at award (%)",
    "utility_shifting_required": "Utility shifting required",
    "design_change_count": "Design changes",
    "dpr_quality_score": "DPR quality score",
    "avg_annual_rainfall_mm": "Annual rainfall (mm)",
    "political_change_during_exec": "Political change",
    "litigation_flag": "Litigation",
    "covid_affected": "COVID-period flag",
    "avg_payment_delay_days": "Payment delay (days)",
    "inflation_rate_pct": "Inflation (%)",
    "material_price_index": "Material price index",
    "delay_months": "Delay (months)",
    "time_overrun_pct": "Time overrun (%)",
    "cost_overrun_pct": "Cost overrun (%)",
    "major_delay": "Major delay (>18 m)",
}


def pretty(col):
    """Return a readable label for a column name."""
    return PRETTY.get(col, col.replace("_", " ").capitalize())
