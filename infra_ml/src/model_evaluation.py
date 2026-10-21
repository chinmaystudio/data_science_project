"""
model_evaluation.py
-------------------
STEP 7 - Measure how good each model is, on the TEST data it never saw.

Classification metrics
  Accuracy   : share of all predictions that are correct
  Precision  : of projects predicted "major delay", how many really were
  Recall     : of real major delays, how many the model caught
  F1-score   : balance of precision and recall
  ROC-AUC    : ability to rank delayed projects above others (0.5 = random)
  Confusion matrix, ROC curve, Precision-Recall curve

Regression metrics
  MAE  : average absolute error (percentage points)
  RMSE : like MAE but punishes big errors more
  R2   : share of the variation explained by the model (1 = perfect)
  Actual-vs-predicted plot, residual plots
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)

from src.eda import MAIN, ACCENT, WARN, PALETTE

# ---------------------------------------------------------------- metrics
def classification_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-score": round(f1_score(y_true, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
    }


def regression_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def baseline_classification(y_train, y_test):
    """A 'model' that always predicts the most common class."""
    majority = int(y_train.mode()[0])
    pred = np.full(len(y_test), majority)
    prob = np.full(len(y_test), y_train.mean())
    return {"Accuracy": round(accuracy_score(y_test, pred), 4), "Precision": 0.0,
            "Recall": 0.0, "F1-score": 0.0, "ROC-AUC": 0.5}


def baseline_regression(y_train, y_test):
    """A 'model' that always predicts the training average."""
    return regression_metrics(y_test, np.full(len(y_test), y_train.mean()))


# ---------------------------------------------------------------- classification charts
def confusion_matrix_plot(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v:,}\n({p:.1f}%)" for v, p in zip(r1, r2)]
                       for r1, r2 in zip(cm, pct)])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                linewidths=2, linecolor="white", annot_kws={"size": 11})
    ax.set(xlabel="Predicted", ylabel="Actual", title=title,
           xticklabels=["No major delay", "Major delay"],
           yticklabels=["No major delay", "Major delay"])
    fig.tight_layout()
    return fig


def roc_curves_plot(y_true, probs):
    """ROC curves of all classification models on one chart."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, prob), color in zip(probs.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, lw=2.2, color=color,
                label=f"{name} (AUC = {roc_auc_score(y_true, prob):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Random guess (AUC = 0.5)")
    ax.set(title="ROC curve", xlabel="False positive rate",
           ylabel="True positive rate (recall)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def precision_recall_plot(y_true, probs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, prob), color in zip(probs.items(), PALETTE):
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ax.plot(recall, precision, lw=2.2, color=color, label=name)
    ax.axhline(np.mean(y_true), ls="--", color="grey", label="Random guess")
    ax.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def probability_histogram(y_true, prob, title):
    """How the predicted probabilities differ for the two real classes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(x=prob, hue=np.where(y_true == 1, "Major delay", "No major delay"),
                 bins=40, stat="density", common_norm=False, element="step",
                 palette={"Major delay": WARN, "No major delay": MAIN}, ax=ax)
    ax.axvline(0.5, color="black", ls="--", lw=1, label="Threshold 0.5")
    ax.set(title=title, xlabel="Predicted probability of major delay", ylabel="Density")
    fig.tight_layout()
    return fig


def metric_comparison_plot(table, metrics, title):
    """Grouped bar chart comparing models on several metrics."""
    long = table.melt(id_vars="model", value_vars=metrics, var_name="metric",
                      value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.barplot(data=long, x="metric", y="score", hue="model", palette=PALETTE[:table["model"].nunique()], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    ax.set(title=title, xlabel="", ylabel="Score")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- regression charts
def actual_vs_predicted_plot(y_true, y_pred, title, sample_size=5000):
    idx = np.random.RandomState(0).choice(len(y_true), min(sample_size, len(y_true)), replace=False)
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(yp, yt, s=8, alpha=0.3, color=MAIN)
    lim = [min(yt.min(), yp.min()), np.percentile(yt, 99.5)]
    ax.plot(lim, lim, color=WARN, lw=2, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set(title=title, xlabel="Predicted cost overrun (%)", ylabel="Actual cost overrun (%)")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_plots(y_true, y_pred, title):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    idx = np.random.RandomState(1).choice(len(residuals), min(5000, len(residuals)), replace=False)
    a1.scatter(np.asarray(y_pred)[idx], residuals[idx], s=8, alpha=0.3, color="#5B7DB1")
    a1.axhline(0, color=WARN, lw=2)
    a1.set(title="Residuals vs predicted", xlabel="Predicted (%)", ylabel="Residual (actual - predicted)")
    sns.histplot(residuals, bins=60, color=MAIN, edgecolor="white", ax=a2)
    a2.axvline(0, color=WARN, lw=2)
    a2.set(title="Residual distribution", xlabel="Residual", ylabel="Projects")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- importance
def feature_importance(pipeline, top=15):
    """Importance of each input feature (trees) or |coefficient| (linear)."""
    pre = pipeline.named_steps["preprocess"].named_steps["columns"]
    names = [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, kind = model.feature_importances_, "Importance"
    else:
        values, kind = np.abs(np.ravel(model.coef_)), "|Coefficient| (scaled inputs)"
    imp = pd.DataFrame({"feature": names, "value": values})
    return imp.sort_values("value", ascending=False).head(top), kind


def feature_importance_plot(pipeline, title, top=15):
    from src.config import pretty
    imp, kind = feature_importance(pipeline, top)
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([pretty(f) for f in imp["feature"]], imp["value"], color=MAIN)
    ax.set(title=title, xlabel=kind)
    fig.tight_layout()
    return fig
