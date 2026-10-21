"""Page 1 - Data overview: what the dataset looks like."""
import pandas as pd

from ui_common import (page_header, section, cards, note, require_training,
                       get_raw_data, get_output, st)

page_header("Data Overview", "A first look at the 200,000 project records before any cleaning.",
     tag="Load data")
require_training()

raw = get_raw_data()
summary = get_output("run_summary.json")

cards([
    ("Rows (projects)", f"{raw.shape[0]:,}", "one row = one project"),
    ("Columns", raw.shape[1], f"{summary['numeric_columns']} numeric · {summary['text_columns']} text"),
    ("Missing cells", f"{summary['missing_cells']:,}", f"{summary['missing_cells'] / raw.size * 100:.2f}% of all cells"),
    ("Duplicate rows", summary["duplicate_rows"], "exact copies"),
])

tab1, tab2, tab3 = st.tabs(["Sample records", "Columns & data types", "Targets"])

with tab1:
    n = st.slider("Number of rows to show", 5, 50, 10)
    st.dataframe(raw.head(n), width="stretch")
    st.caption("df.head() - the first rows of the raw data.")

with tab2:
    types = pd.DataFrame({
        "column": raw.columns,
        "pandas dtype": raw.dtypes.astype(str).values,
        "kind": ["Numerical" if pd.api.types.is_numeric_dtype(raw[c]) else
                 "Date (stored as text)" if "date" in c else "Categorical / text"
                 for c in raw.columns],
        "unique values": [raw[c].nunique() for c in raw.columns],
        "missing": raw.isna().sum().values,
        "example": [str(raw[c].iloc[0]) for c in raw.columns],
    })
    kind = st.multiselect("Filter by kind", types["kind"].unique(), default=list(types["kind"].unique()))
    st.dataframe(types[types["kind"].isin(kind)], width="stretch", height=520)

with tab3:
    section("What we predict")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classification target: `major_delay`**")
        st.write("1 if `delay_months` > 18, else 0. Created in the cleaning step.")
        st.metric("Projects with major delay", f"{(raw['delay_months'] > 18).mean() * 100:.2f}%")
    with c2:
        st.markdown("**Regression target: `cost_overrun_pct`**")
        st.write("(revised cost − sanctioned cost) / sanctioned cost × 100")
        st.metric("Average cost overrun", f"{raw['cost_overrun_pct'].mean():.2f}%")
    note("<b>Why not predict 'delayed or not'?</b> In this dataset whether a project is late "
         "at all is almost random (a test model scored ROC-AUC 0.55 ≈ coin toss), while "
         "<b>major</b> delays follow clear patterns. So the classifier predicts major delays.")
    section("Data origin")
    st.dataframe(raw["data_origin"].value_counts().rename("projects"), width="stretch")
