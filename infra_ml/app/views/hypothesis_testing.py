"""Page 4 - Hypothesis tests and probability distributions (src/hypothesis_tests.py)."""
from ui_common import (page_header, section, note, require_training, show,
                       get_clean_data, st)
from src import hypothesis_tests as ht
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, pretty

page_header("Hypothesis Testing & Probability Distributions",
     "Is a difference real or just chance? SciPy tests at significance level α = 0.05.",
     tag="src/hypothesis_tests.py")
require_training()
df = get_clean_data()


def result_box(r):
    colour = "good" if r["decision"].startswith("Reject") else "note"
    st.markdown(f"**H₀:** {r['H0']}  \n**H₁:** {r['H1']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test statistic", r["statistic"])
    c2.metric("p-value", r["p_value"])
    c3.metric("Decision (α = 0.05)", "Reject H₀" if r["decision"].startswith("Reject") else "Keep H₀")
    note(f"<b>{r['decision']}.</b> {r['interpretation']}", colour)


section("Summary of the five standard tests")
st.dataframe(ht.run_standard_tests(df), width="stretch", hide_index=True)
note("Rule: <b>p-value &lt; 0.05 → reject H₀</b>. With 200,000 rows even tiny differences "
     "become 'significant', so always look at the <b>size</b> of the difference too.")

tabs = st.tabs(["t-test", "Chi-square", "ANOVA", "Correlation test", "Normality",
                "Probability distributions"])
FLAGS = ["litigation_flag", "political_change_during_exec", "utility_shifting_required",
         "covid_affected", "major_delay"]

with tabs[0]:
    st.caption("Compares the MEAN of a numeric variable between two groups.")
    c1, c2 = st.columns(2)
    num = c1.selectbox("Numeric variable", ["cost_overrun_pct", "delay_months"] + NUMERIC_FEATURES,
                       format_func=pretty, key="t_num")
    flag = c2.selectbox("Two groups defined by", FLAGS, format_func=pretty, key="t_flag")
    r = ht.t_test(df, num, flag)
    result_box(r)
    show(ht.t_test_plot(r, num))

with tabs[1]:
    st.caption("Checks whether two CATEGORICAL variables are related.")
    cat = st.selectbox("Category vs major delay", CATEGORICAL_FEATURES + ["contractor_class"],
                       index=CATEGORICAL_FEATURES.index("env_clearance"), format_func=pretty)
    r = ht.chi_square_test(df, cat)
    result_box(r)
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(r["table"], width="stretch")
    with c2:
        show(ht.chi_square_plot(r))

with tabs[2]:
    st.caption("Compares the MEANS of three or more groups.")
    c1, c2 = st.columns(2)
    num = c1.selectbox("Numeric variable", ["cost_overrun_pct", "delay_months"], format_func=pretty, key="a_num")
    cat = c2.selectbox("Groups", CATEGORICAL_FEATURES, format_func=pretty, key="a_cat")
    result_box(ht.anova_test(df, num, cat))

with tabs[3]:
    c1, c2 = st.columns(2)
    x = c1.selectbox("Variable X", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("design_change_count"),
                     format_func=pretty)
    y = c2.selectbox("Variable Y", ["cost_overrun_pct", "delay_months"], format_func=pretty)
    result_box(ht.correlation_test(df, x, y))

with tabs[4]:
    col = st.selectbox("Variable", ["cost_overrun_pct", "delay_months", "inflation_rate_pct",
                                    "bid_to_estimate_ratio"], format_func=pretty)
    r = ht.normality_test(df, col)
    result_box(r)
    show(ht.qq_plot(r, col))
    st.caption("Q-Q plot: if the points follow the red line, the data are close to normal.")

def dist_row(fig, info):
    """Chart on the left, theory-vs-data numbers on the right."""
    c1, c2 = st.columns([2, 1])
    with c1:
        show(fig)
    c2.json(info)


with tabs[5]:
    st.markdown("**Normal distribution: inflation rate**")
    fig, info = ht.fit_normal(df)
    dist_row(fig, info)
    st.markdown("**Poisson distribution: number of design changes (a count)**")
    fig, info = ht.fit_poisson(df)
    dist_row(fig, info)
    st.markdown("**Exponential distribution: delay length of delayed projects**")
    fig, info = ht.fit_exponential(df)
    dist_row(fig, info)
    st.markdown("**Binomial distribution: major delays in a batch of projects**")
    n = st.slider("Projects in the batch (n)", 5, 30, 10)
    fig, info = ht.binomial_example(df, n)
    dist_row(fig, info)
    note("Comparing the theoretical probabilities with the real shares shows how well each "
         "distribution describes the data (e.g. the exponential model under-estimates long "
         "delays when the data have a heavier tail).")
