"""Page 2 - Data cleaning and preprocessing steps and their results."""
import matplotlib.pyplot as plt
import seaborn as sns

from ui_common import (page_header, section, cards, note, require_training, show,
                       get_clean_data, get_output, st)
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, LEAKAGE_COLUMNS
from src.preprocessing import preprocessing_preview, FEATURES
from src.eda import MAIN, WARN

page_header("Cleaning & Preprocessing",
     "From raw records to model-ready numbers: duplicates, invalid values, missing "
     "values, encoding, scaling and the train/test split.",
     tag="src/data_cleaning.py + src/preprocessing.py")
require_training()

df = get_clean_data()
summary = get_output("run_summary.json")
missing = get_output("missing_before.csv").rename(columns={"Unnamed: 0": "column"})
filled = get_output("median_filled.csv").rename(columns={"Unnamed: 0": "column"})
invalid = get_output("invalid_checks.csv")

cards([
    ("Duplicates removed", summary["duplicates_removed"], "df.drop_duplicates()"),
    ("Columns with gaps", len(missing), "before cleaning"),
    ("Rows after cleaning", f"{summary['rows_after_cleaning']:,}", "no rows deleted"),
    ("Train / test", f"{summary['train_rows']:,} / {summary['test_rows']:,}", "80 % / 20 % split"),
])

t1, t2, t3, t4 = st.tabs(["Cleaning", "Missing values", "Preprocessing", "Train/test split"])

with t1:
    section("Validity checks")
    st.dataframe(invalid, width="stretch", hide_index=True)
    note("No impossible values (negative cost, land % above 100 ...) were found. Two "
         "<b>logical</b> issues come from the synthetic data: projects awarded after the "
         "status date, and not-started projects that already show a delay. They are "
         "reported and kept.")
    section("Other cleaning steps")
    st.markdown("- Text dates converted with `pd.to_datetime()`\n"
                "- Extra spaces removed from text with `.str.strip()`\n"
                "- Category `None` (no monitoring) kept as a real value: pandas would "
                "otherwise read the word *None* as missing, so files are read with "
                "`keep_default_na=False`\n"
                "- New target column `major_delay = (delay_months > 18)`")

with t2:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        bars = ax.barh(missing["column"], missing["missing_pct"], color=[WARN] + [MAIN] * (len(missing) - 1))
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
        ax.invert_yaxis()
        ax.set(title="Missing values before cleaning", xlabel="% of rows")
        fig.tight_layout()
        show(fig)
    with c2:
        st.markdown("**Treatment**")
        st.markdown("- **actual_completion_date** - empty because the project is "
                    "*not finished yet*; kept empty (meaningful).\n"
                    "- **Numeric predictors** - filled with the **median** (not 0, and "
                    "not the mean, because the data are skewed).")
        st.dataframe(filled, width="stretch", hide_index=True)

with t3:
    section("Feature selection")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Numeric inputs ({len(NUMERIC_FEATURES)})**")
    c1.write(", ".join(NUMERIC_FEATURES))
    c2.markdown(f"**Categorical inputs ({len(CATEGORICAL_FEATURES)})**")
    c2.write(", ".join(CATEGORICAL_FEATURES))
    c3.markdown("**Excluded (data leakage)**")
    c3.write(", ".join(LEAKAGE_COLUMNS))
    note("<b>Data leakage:</b> revised cost, delay months, status and completion dates "
         "are only known <i>after</i> the project, or are used to build the targets. "
         "Using them as inputs would give fake, too-good results.", "good")
    section("Transformations")
    st.markdown("1. `log10(sanctioned cost)` - cost is extremely skewed (2 to 120,000 crore)\n"
                "2. `StandardScaler` - each numeric column → mean 0, standard deviation 1\n"
                "3. `OneHotEncoder` - each category → its own 0/1 column (first level dropped)")
    before, after = preprocessing_preview(df)
    st.markdown("**Before** (first 5 projects)")
    st.dataframe(before, width="stretch")
    st.markdown(f"**After** - {after.shape[1]} model columns (first 12 shown)")
    st.dataframe(after.iloc[:, :12], width="stretch")
    col = st.selectbox("See the effect of scaling on a column", NUMERIC_FEATURES, index=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    raw_vals = df[col].sample(20000, random_state=0)
    from src.preprocessing import build_preprocessor
    pre = build_preprocessor().fit(df[FEATURES])
    scaled = pre.transform(df[FEATURES].sample(20000, random_state=0))[:, NUMERIC_FEATURES.index(col)]
    sns.histplot(raw_vals, bins=50, color=MAIN, ax=a1); a1.set_title("Original values")
    sns.histplot(scaled, bins=50, color="#F28C28", ax=a2); a2.set_title("After preprocessing (mean 0, sd 1)")
    fig.tight_layout()
    show(fig)

with t4:
    st.write("`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.pie([summary["train_rows"], summary["test_rows"]], labels=["Training 80%", "Testing 20%"],
               colors=[MAIN, "#F28C28"], autopct=lambda p: f"{p * len(df) / 100:,.0f}",
               wedgeprops=dict(edgecolor="white"), startangle=90)
        ax.set_title("Train / test split")
        show(fig)
    with c2:
        st.markdown("- Models **learn** only from the training set.\n"
                    "- They are **scored** only on the test set (projects they never saw).\n"
                    "- `stratify=y` keeps the same share of major delays (≈18 %) in both parts.\n"
                    "- `random_state=42` makes the split identical every run.")
