"""
model_training.py
-----------------
STEP 6 - Train classification and regression models with scikit-learn.

Classification (target = major_delay, 1 if delay > 18 months)
  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
Regression (target = cost_overrun_pct)
  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor

Each model is a Pipeline = preprocessing (scaling + encoding) + algorithm,
so new data is processed exactly like the training data.

class_weight="balanced": only about 18 % of projects have a major delay.
Without balancing, a model can score high accuracy by almost always saying
"no delay"; balancing gives the minority class more weight so the model
learns to find major delays (higher recall).
"""
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import MODEL_DIR, RANDOM_STATE
from src.preprocessing import build_preprocessor


def classification_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    }


def regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=50, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE),
    }


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def train_models(X_train, y_train, task="classification"):
    """Fit every model and return {name: fitted pipeline} and timing table."""
    models = classification_models() if task == "classification" else regression_models()
    fitted, timing = {}, []
    for name, model in models.items():
        start = time.time()
        pipe = make_pipeline(model).fit(X_train, y_train)
        fitted[name] = pipe
        timing.append({"model": name, "train_seconds": round(time.time() - start, 1)})
        print(f"  trained {task:<14} {name:<20} in {timing[-1]['train_seconds']} s")
    return fitted, pd.DataFrame(timing)


def cross_validate(X_train, y_train, task="classification", sample_size=30000, folds=5):
    """5-fold cross-validation on a random sample (keeps it fast).

    Classification is scored with ROC-AUC, regression with R-squared.
    The spread (std) across folds shows how stable each model is."""
    sample = X_train.sample(min(sample_size, len(X_train)), random_state=RANDOM_STATE)
    y = y_train.loc[sample.index]
    models = classification_models() if task == "classification" else regression_models()
    scoring = "roc_auc" if task == "classification" else "r2"
    rows = []
    for name, model in models.items():
        scores = cross_val_score(make_pipeline(model), sample, y, cv=folds,
                                 scoring=scoring, n_jobs=-1)
        rows.append({"model": name, "metric": scoring, "cv_mean": round(scores.mean(), 4),
                     "cv_std": round(scores.std(), 4),
                     **{f"fold_{i + 1}": round(s, 4) for i, s in enumerate(scores)}})
    return pd.DataFrame(rows)


def save_models(fitted, task):
    MODEL_DIR.mkdir(exist_ok=True)
    for name, pipe in fitted.items():
        path = MODEL_DIR / f"{task}_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipe, path, compress=3)


def load_models(task):
    """Load every saved model for a task -> {name: pipeline}."""
    models = {}
    for path in sorted(MODEL_DIR.glob(f"{task}_*.joblib")):
        name = path.stem.replace(f"{task}_", "").replace("_", " ").title()
        models[name] = joblib.load(path)
    return models
