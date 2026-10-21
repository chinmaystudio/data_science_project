"""
train.py
--------
Run the whole pipeline ONCE before opening the UI:

    python train.py

  1. load data          (src/data_loader.py)
  2. clean data         (src/data_cleaning.py)
  3. split + preprocess (src/preprocessing.py)
  4. train models       (src/model_training.py)
  5. evaluate on test   (src/model_evaluation.py)
  6. save models (models/) and result tables (outputs/)
"""
import json

import pandas as pd

from src.config import OUTPUT_DIR, MODEL_DIR
from src.data_loader import load_raw_data, describe_dataset
from src.data_cleaning import clean_data
from src.preprocessing import split_data
from src.model_training import train_models, cross_validate, save_models
from src.model_evaluation import (classification_metrics, regression_metrics,
                                  baseline_classification, baseline_regression)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("1) Loading data ...")
    raw = load_raw_data()
    info = describe_dataset(raw)
    print("  ", info)

    print("2) Cleaning data ...")
    df, report = clean_data(raw)
    report["invalid_checks"].to_csv(OUTPUT_DIR / "invalid_checks.csv", index=False)
    report["missing_before"].to_csv(OUTPUT_DIR / "missing_before.csv")
    report["median_filled"].to_csv(OUTPUT_DIR / "median_filled.csv")
    summary = {**info, "duplicates_removed": report["duplicates_removed"],
               "rows_after_cleaning": report["rows_after_cleaning"],
               "major_delay_rate_pct": report["major_delay_rate_pct"]}

    # ------------------------------------------------ classification
    print("3) Classification: major delay (> 18 months)")
    X_tr, X_te, y_tr, y_te = split_data(df, "classification")
    summary.update(train_rows=len(X_tr), test_rows=len(X_te))
    models, timing = train_models(X_tr, y_tr, "classification")
    save_models(models, "classification")
    rows = [{"model": "Baseline (always 'no major delay')",
             **baseline_classification(y_tr, y_te)}]
    preds = pd.DataFrame({"actual": y_te.values})
    for name, pipe in models.items():
        prob = pipe.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"model": name, **classification_metrics(y_te, pred, prob)})
        preds[f"prob_{name}"] = prob
    pd.DataFrame(rows).merge(timing, on="model", how="left").to_csv(
        OUTPUT_DIR / "classification_metrics.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "classification_test_predictions.csv", index=False)
    print("   cross-validation ...")
    cross_validate(X_tr, y_tr, "classification").to_csv(
        OUTPUT_DIR / "classification_cv.csv", index=False)

    # ------------------------------------------------ regression
    print("4) Regression: cost overrun (%)")
    X_tr, X_te, y_tr, y_te = split_data(df, "regression")
    models, timing = train_models(X_tr, y_tr, "regression")
    save_models(models, "regression")
    rows = [{"model": "Baseline (always predict mean)", **baseline_regression(y_tr, y_te)}]
    preds = pd.DataFrame({"actual": y_te.values})
    for name, pipe in models.items():
        pred = pipe.predict(X_te)
        rows.append({"model": name, **regression_metrics(y_te, pred)})
        preds[f"pred_{name}"] = pred
    pd.DataFrame(rows).merge(timing, on="model", how="left").to_csv(
        OUTPUT_DIR / "regression_metrics.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "regression_test_predictions.csv", index=False)
    print("   cross-validation ...")
    cross_validate(X_tr, y_tr, "regression").to_csv(
        OUTPUT_DIR / "regression_cv.csv", index=False)

    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nClassification results:")
    print(pd.read_csv(OUTPUT_DIR / "classification_metrics.csv").to_string(index=False))
    print("\nRegression results:")
    print(pd.read_csv(OUTPUT_DIR / "regression_metrics.csv").to_string(index=False))
    print("\nDone. Now start the UI with:  streamlit run app/main.py")


if __name__ == "__main__":
    main()
