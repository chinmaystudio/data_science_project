"""Page 5 - Model evaluation on the 40,000 test projects (src/model_evaluation.py)."""
import numpy as np

from ui_common import (page_header, section, cards, note, require_training, show,
                       get_models, get_output, st)
from src import model_evaluation as me

page_header("Model Performance",
     "Every model is scored on test projects it never saw during training.",
     tag="src/model_training.py + src/model_evaluation.py")
require_training()

clf_tab, reg_tab = st.tabs(["Classification · major delay", "Regression · cost overrun %"])

# ------------------------------------------------------------------ classification
with clf_tab:
    metrics = get_output("classification_metrics.csv")
    preds = get_output("classification_test_predictions.csv")
    cv = get_output("classification_cv.csv")
    models = get_models("classification")
    best = metrics.iloc[1:].sort_values("ROC-AUC", ascending=False).iloc[0]
    cards([("Best model", best["model"], "highest ROC-AUC"),
           ("ROC-AUC", f"{best['ROC-AUC']:.3f}", "0.5 = random guess"),
           ("Recall", f"{best['Recall'] * 100:.1f}%", "major delays caught"),
           ("Accuracy", f"{best['Accuracy'] * 100:.1f}%", "all predictions")])
    section("Metrics on the test set")
    st.dataframe(metrics.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                                             color="#FFF2CC"), width="stretch", hide_index=True)
    note("The <b>baseline</b> that always says 'no major delay' already gets ~82% accuracy but "
         "catches <b>zero</b> delays. That is why accuracy alone is misleading for imbalanced "
         "data - look at recall, F1 and ROC-AUC.")
    show(me.metric_comparison_plot(metrics.iloc[1:], ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                                   "Classification models compared"))
    y = preds["actual"].values
    probs = {name: preds[f"prob_{name}"].values for name in metrics["model"].iloc[1:]}
    c1, c2 = st.columns(2)
    with c1: show(me.roc_curves_plot(y, probs))
    with c2: show(me.precision_recall_plot(y, probs))

    section("Look inside one model")
    name = st.selectbox("Model", list(probs.keys()), index=list(probs.keys()).index(best["model"]))
    threshold = st.slider("Decision threshold (predict 'major delay' if probability ≥ threshold)",
                          0.10, 0.90, 0.50, 0.05)
    pred = (probs[name] >= threshold).astype(int)
    m = me.classification_metrics(y, pred, probs[name])
    cards([(k, f"{v:.3f}", "") for k, v in m.items()])
    c1, c2 = st.columns([1, 1.25])
    with c1: show(me.confusion_matrix_plot(y, pred, f"Confusion matrix · {name}"))
    with c2: show(me.probability_histogram(y, probs[name], f"Predicted probabilities · {name}"))
    st.caption("Lower threshold → more delays caught (recall ↑) but more false alarms (precision ↓).")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        show(me.feature_importance_plot(models[name], f"Top features · {name}"))
    c2.markdown("**How to read this**")
    c2.markdown("- Tree models: *importance* = how much a feature helps the splits.\n"
                "- Logistic Regression: size of the coefficient on scaled inputs.\n"
                "- Important ≠ causes: it only shows what the model relies on.")

    section("5-fold cross-validation (ROC-AUC, 30,000-project sample)")
    st.dataframe(cv, width="stretch", hide_index=True)

# ------------------------------------------------------------------ regression
with reg_tab:
    metrics = get_output("regression_metrics.csv")
    preds = get_output("regression_test_predictions.csv")
    cv = get_output("regression_cv.csv")
    models = get_models("regression")
    best = metrics.iloc[1:].sort_values("R2", ascending=False).iloc[0]
    base = metrics.iloc[0]
    cards([("Best model", best["model"], "highest R²"),
           ("R²", f"{best['R2']:.3f}", "share of variation explained"),
           ("RMSE", f"{best['RMSE']:.2f} pts", f"baseline {base['RMSE']:.2f}"),
           ("MAE", f"{best['MAE']:.2f} pts", f"baseline {base['MAE']:.2f}")])
    st.dataframe(metrics.style.highlight_max(subset=["R2"], color="#FFF2CC")
                 .highlight_min(subset=["MAE", "RMSE"], color="#FFF2CC"),
                 width="stretch", hide_index=True)
    show(me.metric_comparison_plot(metrics.iloc[1:], ["MAE", "RMSE"], "Error of each model (lower is better)"))
    note("Here the simple <b>Linear Regression</b> beats the Random Forest: the relationships in "
         "this dataset are mostly straight-line, so a more complex model adds no benefit.", "good")
    name = st.selectbox("Model", [m for m in metrics["model"].iloc[1:]],
                        index=list(metrics["model"].iloc[1:]).index(best["model"]), key="reg_model")
    y, p = preds["actual"].values, preds[f"pred_{name}"].values
    c1, c2 = st.columns([1, 1.2])
    with c1: show(me.actual_vs_predicted_plot(y, p, f"Actual vs predicted · {name}"))
    with c2: show(me.feature_importance_plot(models[name], f"Top features · {name}"))
    show(me.residual_plots(y, p, f"Residual analysis · {name}"))
    st.caption("Good residuals are centred on 0 with no pattern. A wide spread or long right "
               "tail means very large overruns are hard to predict.")
    section("5-fold cross-validation (R², 30,000-project sample)")
    st.dataframe(cv, width="stretch", hide_index=True)
