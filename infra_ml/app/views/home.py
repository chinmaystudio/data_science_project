"""Home page of the portal."""
import pandas as pd

from ui_common import (page_header, section, cards, note, require_training,
                       get_output, st)

page_header("Welcome to the Prediction Portal",
            "Early warning of major delays and cost overruns in infrastructure projects, "
            "using exploratory data analysis, statistical tests and machine-learning models.")

require_training()
summary = get_output("run_summary.json")
clf = get_output("classification_metrics.csv")
reg = get_output("regression_metrics.csv")
best_clf = clf.iloc[1:].sort_values("ROC-AUC", ascending=False).iloc[0]
best_reg = reg.iloc[1:].sort_values("R2", ascending=False).iloc[0]

cards([
    ("Projects in dataset", f"{summary['rows']:,}", f"{summary['columns']} columns"),
    ("Major delays (> 18 months)", f"{summary['major_delay_rate_pct']}%", "share of all projects"),
    ("Best classifier (ROC-AUC)", f"{best_clf['ROC-AUC']:.3f}", best_clf["model"]),
    ("Best regressor (R²)", f"{best_reg['R2']:.3f}", best_reg["model"]),
])

left, right = st.columns([1.6, 1], gap="large")

with left:
    section("About the Portal")
    st.write(
        "Infrastructure projects often finish late and cost more than sanctioned. "
        "Like churn prediction in telecom or predictive maintenance in manufacturing, "
        "this portal uses historical project records to identify projects that are "
        "likely to go wrong, so that corrective action can be taken before work starts.")
    section("Prediction Tasks")
    st.table(pd.DataFrame({
        "Task": ["Classification", "Regression"],
        "Question answered": ["Will the project suffer a major delay (more than 18 months)?",
                              "By what percentage will the cost exceed the sanctioned cost?"],
        "Target column": ["major_delay", "cost_overrun_pct"],
        "Models": ["Logistic Regression, Decision Tree, Random Forest",
                   "Linear Regression, Decision Tree, Random Forest"],
    }).set_index("Task"))
    section("Workflow")
    st.table(pd.DataFrame({
        "Step": [str(i) for i in range(1, 7)],
        "Page": ["Data Overview", "Cleaning & Preprocessing", "EDA", "Hypothesis Testing",
                 "Model Performance", "Predict"],
        "Contents": ["Rows, columns, data types, sample records",
                     "Duplicates, invalid values, missing values, encoding, scaling, train/test split",
                     "Summary statistics, distributions, outliers, correlation analysis",
                     "t-test, chi-square, ANOVA, correlation, normality, probability distributions",
                     "Metrics, confusion matrix, ROC curve, residuals, feature importance, cross-validation",
                     "Enter a new project and obtain the predictions"],
    }).set_index("Step"))

with right:
    section("Quick Links")
    for title, path in [("Data Overview", "views/data_overview.py"),
                        ("Exploratory Data Analysis", "views/eda.py"),
                        ("Hypothesis Testing", "views/hypothesis_testing.py"),
                        ("Model Performance", "views/model_performance.py"),
                        ("Predict a New Project", "views/predict.py")]:
        st.page_link(path, label=f"» {title}")
    section("Important Notice")
    note("Every record in the dataset is marked <b>SYNTHETIC</b> (data_origin column). "
         "Sector, state, authority and cost patterns come from real archive records, but the "
         "risk factors and outcomes were simulated. Results demonstrate the method and are "
         "not real government project outcomes.")
    section("Dataset at a Glance")
    st.table(pd.DataFrame({
        "Item": ["File", "Sheet", "Rows", "Columns", "Training rows", "Testing rows"],
        "Value": ["NIRIKSHAK_Projects_200000.xlsx", "Projects", f"{summary['rows']:,}",
                  str(summary["columns"]), f"{summary['train_rows']:,}", f"{summary['test_rows']:,}"],
    }).set_index("Item"))
