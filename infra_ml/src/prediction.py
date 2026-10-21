"""
prediction.py
-------------
STEP 8 - Use the trained models to predict a NEW project.

The UI collects the project details into a dictionary; this file turns
it into a one-row DataFrame, runs the saved pipelines and returns:
  * probability of a major delay (> 18 months) and the predicted class
  * predicted cost overrun (%) and the predicted revised cost
  * a simple list of risk flags (which inputs are worse than typical)
"""
import pandas as pd

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def to_frame(project):
    """Dictionary of inputs -> one-row DataFrame in the right column order."""
    return pd.DataFrame([{col: project[col] for col in FEATURES}])


def predict_project(project, classifier, regressor, threshold=0.5):
    row = to_frame(project)
    prob = float(classifier.predict_proba(row)[0, 1])
    overrun = float(regressor.predict(row)[0])
    cost = project["sanctioned_cost_inr_cr"]
    return {
        "major_delay_probability": prob,
        "major_delay_predicted": prob >= threshold,
        "cost_overrun_pct": overrun,
        "revised_cost_inr_cr": cost * (1 + overrun / 100),
        "extra_cost_inr_cr": cost * overrun / 100,
    }


def risk_level(prob):
    if prob < 0.30:
        return "Low", "#2E7D32"
    if prob < 0.55:
        return "Medium", "#C88A00"
    return "High", "#B22222"


def risk_flags(project, reference):
    """Compare inputs with the dataset (reference) and list warning signs."""
    flags = []
    if project["land_acquired_pct_at_award"] < 70:
        flags.append("Less than 70 % of land acquired at award")
    if project["env_clearance"] == "Pending":
        flags.append("Environmental clearance pending")
    if project["forest_clearance"] == "Pending":
        flags.append("Forest clearance pending")
    if project["litigation_flag"] == 1:
        flags.append("Litigation / court case")
    if project["political_change_during_exec"] == 1:
        flags.append("Change of government expected during execution")
    if project["design_change_count"] >= 3:
        flags.append(f"{project['design_change_count']} design changes")
    if project["bid_to_estimate_ratio"] < 0.85:
        flags.append("Winning bid far below the engineer's estimate")
    if project["dpr_quality_score"] < 5:
        flags.append("Low DPR (project report) quality")
    if project["complexity_index"] >= 0.6:
        flags.append("High technical complexity")
    if project["terrain"] in ("Hilly", "Mountainous"):
        flags.append(f"{project['terrain']} terrain")
    if project["avg_payment_delay_days"] > reference["avg_payment_delay_days"].quantile(0.75):
        flags.append("Slow payments to the contractor")
    if project["material_price_index"] > 1.1:
        flags.append("Material prices above estimate")
    return flags


def compare_with_dataset(project, reference):
    """Project value vs dataset median for the main numeric inputs."""
    cols = ["land_acquired_pct_at_award", "design_change_count", "bid_to_estimate_ratio",
            "contractor_experience_yrs", "dpr_quality_score", "avg_payment_delay_days",
            "complexity_index", "planned_duration_months"]
    return pd.DataFrame({"This project": [project[c] for c in cols],
                         "Dataset median": [reference[c].median() for c in cols]},
                        index=cols)
