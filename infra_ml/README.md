# Infrastructure Project Delay & Cost Overrun Predictor

A Python data-science mini-project (predictive analytics case study) on
200,000 infrastructure projects. It predicts:

* **Classification**: will a project suffer a **major delay** (more than 18 months)?
  Models: Logistic Regression, Decision Tree, Random Forest
* **Regression**: what will the **cost overrun (%)** be?
  Models: Linear Regression, Decision Tree, Random Forest

Libraries used: pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn and Streamlit (for the UI).

> **Data note:** every record in `NIRIKSHAK_Projects_200000.xlsx` is marked
> *SYNTHETIC* (`data_origin` column). The sector, state, authority and cost
> patterns come from real archive records, but the risk factors and outcomes
> were simulated. The results demonstrate the method; they are not real
> government outcomes.

---

## How to run

```bash
# 1. install the libraries (once)
pip install -r requirements.txt

# 2. clean the data, train and evaluate all models (once, about 4 minutes)
python train.py

# 3. open the user interface in the browser
streamlit run app/main.py
```

Run all commands from this project folder. The Excel file must be in `data/`.

---

## Project structure: one file per step

```
infra_ml/
├── data/
│   └── NIRIKSHAK_Projects_200000.xlsx   input dataset
├── src/                                  ML pipeline (no UI code)
│   ├── config.py           paths, feature lists, targets
│   ├── data_loader.py      STEP 1  load the Excel file (+ fast CSV copy)
│   ├── data_cleaning.py    STEP 2  duplicates, invalid values, missing values, target
│   ├── preprocessing.py    STEP 3  log transform, scaling, one-hot encoding, train/test split
│   ├── eda.py              STEP 4  summary statistics + all Matplotlib/Seaborn charts
│   ├── hypothesis_tests.py STEP 5  t-test, chi-square, ANOVA, correlation, normality,
│   │                               Normal / Poisson / Exponential / Binomial distributions
│   ├── model_training.py   STEP 6  the 6 scikit-learn models + cross-validation
│   ├── model_evaluation.py STEP 7  metrics, confusion matrix, ROC, residuals, importance
│   └── prediction.py       STEP 8  predict a new project + risk flags
├── train.py                runs steps 1-3 and 6-7, saves models/ and outputs/
├── app/                                  UI (Streamlit), separate from the pipeline
│   ├── main.py             entry point: portal header, menu bar, footer
│   ├── ui_common.py        portal style (CSS), cached loading, statistic boxes
│   └── views/              one file per page
│       ├── home.py
│       ├── data_overview.py
│       ├── cleaning_preprocessing.py
│       ├── eda.py
│       ├── hypothesis_testing.py
│       ├── model_performance.py
│       └── predict.py
├── models/                 saved trained models (.joblib), created by train.py
├── outputs/                metric tables and test predictions, created by train.py
├── .streamlit/config.toml  UI colour theme (flat portal colours)
└── requirements.txt
```

You can also run each pipeline file on its own to see its output, e.g.
`python -m src.data_cleaning` or `python -m src.hypothesis_tests`.

---

## Syllabus coverage

| Topic | Where |
|---|---|
| Data visualisation (Matplotlib, Seaborn) | `src/eda.py`, every UI page |
| EDA: summary statistics | `eda.summary_statistics`, `category_summary`, `outlier_table` |
| EDA: distribution analysis | histograms + KDE, boxplots, violin plots, Q-Q plot |
| EDA: correlation analysis | heatmap, correlation with target, scatter + trend line, pair plot |
| Hypothesis testing | t-test, chi-square, one-way ANOVA, Pearson test, normality test |
| Probability distributions | Normal, Poisson, Exponential, Binomial fits (theory vs data) |
| Classification algorithms | Logistic Regression, Decision Tree, Random Forest |
| Regression algorithms | Linear Regression, Decision Tree, Random Forest |
| Model evaluation | accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC & PR curves, MAE, RMSE, R², residual plots, 5-fold cross-validation, baseline comparison |
| scikit-learn hands-on | Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder, train_test_split, cross_val_score |
| Mini-project / case study | Predictive analytics for infrastructure projects (like churn / predictive maintenance) |

---

## Results (test set: 40,000 projects never seen in training)

| Classification: major delay | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "no") | 0.818 | 0 | 0 | 0 | 0.500 |
| **Logistic Regression** | 0.776 | 0.435 | 0.780 | 0.559 | **0.859** |
| Decision Tree | 0.707 | 0.352 | 0.723 | 0.474 | 0.782 |
| Random Forest | 0.757 | 0.409 | 0.761 | 0.532 | 0.837 |

| Regression: cost overrun % | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (predict mean) | 13.03 | 17.78 | 0.000 |
| **Linear Regression** | 10.39 | 14.03 | **0.378** |
| Decision Tree | 11.26 | 15.29 | 0.260 |
| Random Forest | 10.69 | 14.49 | 0.335 |

Key points:
* Accuracy alone is misleading: the baseline gets 82 % accuracy but finds no
  delays. Models use `class_weight="balanced"` and are compared on recall, F1 and ROC-AUC.
* The simple linear models beat Random Forest because the relationships in
  this dataset are mostly linear.
* "Delayed or not" was **not** used as the target: a test model scored only
  ROC-AUC 0.55, so whether a project is late at all is almost random in this data.
