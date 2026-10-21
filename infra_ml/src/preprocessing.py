"""
preprocessing.py
----------------
STEP 3 - Turn clean data into numbers a model can learn from.

  * Feature engineering : log10 of cost (cost is very skewed)
  * Encoding            : categories -> 0/1 columns (One-Hot Encoding)
  * Scaling             : numeric columns -> mean 0, std 1 (StandardScaler)
  * Train / test split  : 80 % train, 20 % test (stratified for classes)

Everything is wrapped in a scikit-learn ColumnTransformer so that exactly
the same steps are applied during training AND when the UI predicts a new
project.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from src.config import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLASS_TARGET,
                        REG_TARGET, RANDOM_STATE, TEST_SIZE)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def log_cost(X):
    """Replace sanctioned cost by log10(cost) - reduces extreme skew."""
    X = X.copy()
    X["sanctioned_cost_inr_cr"] = np.log10(X["sanctioned_cost_inr_cr"])
    return X


def build_preprocessor():
    """Return the preprocessing object (not fitted yet)."""
    numeric_steps = Pipeline([
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
         CATEGORICAL_FEATURES),
    ])
    # log transform first, then scale / encode
    return Pipeline([
        ("log_cost", FunctionTransformer(log_cost)),
        ("columns", preprocessor),
    ])


def split_data(df, task="classification"):
    """Split into X_train, X_test, y_train, y_test for the chosen task."""
    X = df[FEATURES]
    if task == "classification":
        y = df[CLASS_TARGET]
        return train_test_split(X, y, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE, stratify=y)
    y = df[REG_TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def feature_names(fitted_preprocessor):
    """Names of the columns after encoding (for feature-importance charts)."""
    columns = fitted_preprocessor.named_steps["columns"]
    return list(columns.get_feature_names_out())


def preprocessing_preview(df, n=5):
    """Show a few rows before and after preprocessing (used in the UI)."""
    sample = df[FEATURES].head(n)
    pre = build_preprocessor().fit(df[FEATURES])
    after = pd.DataFrame(pre.transform(sample), columns=feature_names(pre)).round(3)
    return sample, after


if __name__ == "__main__":
    from src.data_cleaning import load_clean_data
    data = load_clean_data()
    X_tr, X_te, y_tr, y_te = split_data(data)
    print("Train:", X_tr.shape, " Test:", X_te.shape)
    print(preprocessing_preview(data)[1].iloc[:, :8])
