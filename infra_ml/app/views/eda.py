"""Page 3 - Exploratory Data Analysis (all charts come from src/eda.py)."""
from ui_common import (page_header, section, cards, note, require_training, show,
                       get_clean_data, st)
from src import eda
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, pretty

page_header("Exploratory Data Analysis",
     "Summary statistics, distributions, group comparisons and correlations - "
     "built with Matplotlib and Seaborn.",
     tag="src/eda.py")
require_training()
df = get_clean_data()

cards([
    ("Mean delay", f"{df['delay_months'].mean():.1f} m", f"median {df['delay_months'].median():.1f} m"),
    ("Major-delay rate", f"{df['major_delay'].mean() * 100:.1f}%", "delay > 18 months"),
    ("Mean cost overrun", f"{df['cost_overrun_pct'].mean():.1f}%", f"median {df['cost_overrun_pct'].median():.1f}%"),
    ("Median project cost", f"₹{df['sanctioned_cost_inr_cr'].median():.0f} Cr", f"mean ₹{df['sanctioned_cost_inr_cr'].mean():.0f} Cr"),
])

tabs = st.tabs(["Summary statistics", "Distributions", "Outliers & spread",
                "Group comparison", "Correlation"])

NUMS = NUMERIC_FEATURES + ["delay_months", "cost_overrun_pct"]
CATS = CATEGORICAL_FEATURES + ["state", "contractor_class", "status"]

with tabs[0]:
    section("Numeric summary (describe + median + skewness)")
    st.dataframe(eda.summary_statistics(df), width="stretch", height=560)
    note("When the <b>mean</b> is much larger than the <b>median</b> and skewness is high "
         "(cost, delay, land required), the variable is right-skewed: a few very large values "
         "pull the mean up.")
    cat = st.selectbox("Category summary for", CATS, index=0)
    st.dataframe(eda.category_summary(df, cat), width="stretch")

with tabs[1]:
    show(eda.target_overview(df))
    col = st.selectbox("Histogram of", NUMS, index=NUMS.index("cost_overrun_pct"),
                       format_func=pretty)
    bins = st.slider("Bins", 10, 100, 50)
    show(eda.histogram(df, col, bins=bins))
    section("All numeric inputs at a glance")
    show(eda.histogram_grid(df, NUMERIC_FEATURES[:12]))
    show(eda.projects_per_year(df))

with tabs[2]:
    section("Outliers (1.5 × IQR rule)")
    st.dataframe(eda.outlier_table(df), width="stretch", hide_index=True, height=300)
    c1, c2 = st.columns(2)
    num = c1.selectbox("Numeric variable", ["delay_months", "cost_overrun_pct", "time_overrun_pct"],
                       format_func=pretty)
    cat = c2.selectbox("Group by", CATEGORICAL_FEATURES, index=2, format_func=pretty)
    ylim = (0, 60) if num == "delay_months" else (-20, 90) if num == "cost_overrun_pct" else (0, 150)
    show(eda.boxplot_by_category(df, num, cat, ylim=ylim))
    show(eda.violin_by_category(df, num, cat))

with tabs[3]:
    cat = st.selectbox("Compare major-delay rate across", CATS, index=CATS.index("forest_clearance"),
                       format_func=pretty, key="rate_cat")
    show(eda.rate_by_category(df, cat))
    show(eda.stacked_delay_mix(df, cat if df[cat].nunique() <= 12 else "sector"))
    num = st.selectbox("Major-delay rate across groups of", NUMERIC_FEATURES,
                       index=NUMERIC_FEATURES.index("land_acquired_pct_at_award"), format_func=pretty)
    show(eda.binned_rate(df, num))

with tabs[4]:
    show(eda.correlation_heatmap(df))
    target = st.radio("Correlation of every feature with", ["cost_overrun_pct", "delay_months"],
                      horizontal=True, format_func=pretty)
    show(eda.correlation_with_target(df, target))
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("design_change_count"),
                     format_func=pretty)
    y = c2.selectbox("Y axis", ["cost_overrun_pct", "delay_months", "time_overrun_pct"], format_func=pretty)
    hue = c3.selectbox("Colour by", [None, "sector", "terrain", "env_clearance", "major_delay"],
                       format_func=lambda v: "None" if v is None else pretty(v))
    show(eda.scatter_with_trend(df, x, y, hue))
    note("Correlation measures a <b>linear association</b> only - it does not prove that one "
         "variable causes the other.")
    section("Pair plot (sample of 1,500 projects)")
    show(eda.pair_plot(df, ["land_acquired_pct_at_award", "design_change_count",
                            "bid_to_estimate_ratio", "cost_overrun_pct"]))
