"""
hypothesis_tests.py
-------------------
STEP 5 - Statistical modelling: hypothesis testing and probability
distributions (SciPy).

Every test returns a dictionary with:
  test, H0 (null hypothesis), H1 (alternative), statistic, p_value,
  decision (at alpha = 0.05) and a plain-English interpretation.

Rule: if p-value < 0.05 we REJECT H0 (the difference / relationship is
statistically significant). A significant result shows an association,
not proof of cause.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import pretty
from src.eda import MAIN, ACCENT, WARN, PALETTE

ALPHA = 0.05


def _decision(p):
    return "Reject H0 (significant)" if p < ALPHA else "Fail to reject H0 (not significant)"


def _fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ---------------------------------------------------------------- tests
def t_test(df, numeric_col, flag_col):
    """Welch two-sample t-test: is the mean different for flag 1 vs 0?"""
    g1 = df.loc[df[flag_col] == 1, numeric_col].dropna()
    g0 = df.loc[df[flag_col] == 0, numeric_col].dropna()
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return {
        "test": "Two-sample t-test (Welch)",
        "H0": f"Mean {pretty(numeric_col)} is the same with and without {pretty(flag_col)}",
        "H1": f"Mean {pretty(numeric_col)} is different",
        "statistic": round(float(t), 3), "p_value": _fmt_p(p),
        "decision": _decision(p),
        "interpretation": f"Mean with {pretty(flag_col)} = {g1.mean():.2f}, "
                          f"without = {g0.mean():.2f} (difference {g1.mean() - g0.mean():+.2f}).",
        "groups": {"Yes (1)": g1, "No (0)": g0},
    }


def chi_square_test(df, cat_col, target="major_delay"):
    """Chi-square test of independence between a category and the target."""
    table = pd.crosstab(df[cat_col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    rates = (df.groupby(cat_col)[target].mean() * 100).round(2)
    return {
        "test": "Chi-square test of independence",
        "H0": f"{pretty(cat_col)} and {pretty(target)} are independent",
        "H1": f"{pretty(cat_col)} and {pretty(target)} are associated",
        "statistic": round(float(chi2), 2), "dof": int(dof), "p_value": _fmt_p(p),
        "decision": _decision(p),
        "interpretation": "Major-delay rate by group: " +
                          ", ".join(f"{k}: {v}%" for k, v in rates.items()),
        "table": table,
    }


def anova_test(df, numeric_col, cat_col):
    """One-way ANOVA: are the group means equal across categories?"""
    groups = [g[numeric_col].dropna() for _, g in df.groupby(cat_col)]
    f, p = stats.f_oneway(*groups)
    means = df.groupby(cat_col)[numeric_col].mean().round(2)
    return {
        "test": "One-way ANOVA",
        "H0": f"Mean {pretty(numeric_col)} is equal in every {pretty(cat_col)} group",
        "H1": "At least one group mean is different",
        "statistic": round(float(f), 2), "p_value": _fmt_p(p),
        "decision": _decision(p),
        "interpretation": "Group means: " + ", ".join(f"{k}: {v}" for k, v in means.items()),
    }


def correlation_test(df, x, y):
    """Pearson correlation test: is r different from 0?"""
    d = df[[x, y]].dropna()
    r, p = stats.pearsonr(d[x], d[y])
    strength = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else \
               "moderate" if abs(r) < 0.5 else "strong"
    return {
        "test": "Pearson correlation test",
        "H0": f"No linear correlation between {pretty(x)} and {pretty(y)} (r = 0)",
        "H1": "There is a linear correlation (r != 0)",
        "statistic": round(float(r), 4), "p_value": _fmt_p(p),
        "decision": _decision(p),
        "interpretation": f"r = {r:.3f}: a {strength} "
                          f"{'positive' if r > 0 else 'negative'} linear relationship.",
    }


def normality_test(df, column, sample_size=5000):
    """D'Agostino-Pearson normality test on a random sample."""
    s = df[column].dropna().sample(min(sample_size, len(df)), random_state=0)
    stat, p = stats.normaltest(s)
    return {
        "test": "D'Agostino-Pearson normality test",
        "H0": f"{pretty(column)} follows a normal distribution",
        "H1": f"{pretty(column)} is not normally distributed",
        "statistic": round(float(stat), 2), "p_value": _fmt_p(p),
        "decision": _decision(p),
        "interpretation": f"Skewness = {s.skew():.2f}, kurtosis = {s.kurt():.2f} "
                          "(normal data: both close to 0).",
        "sample": s,
    }


def run_standard_tests(df):
    """The five tests used in the project report, as one table."""
    results = [
        t_test(df, "cost_overrun_pct", "litigation_flag"),
        chi_square_test(df, "env_clearance"),
        anova_test(df, "cost_overrun_pct", "sector"),
        correlation_test(df, "design_change_count", "cost_overrun_pct"),
        normality_test(df, "cost_overrun_pct"),
    ]
    keys = ["test", "H0", "statistic", "p_value", "decision"]
    return pd.DataFrame([{k: r[k] for k in keys} for r in results])


# ---------------------------------------------------------------- charts for tests
def t_test_plot(result, numeric_col):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (label, data), color in zip(result["groups"].items(), [WARN, MAIN]):
        sns.kdeplot(data.clip(upper=data.quantile(0.995)), fill=False, lw=2.2,
                    color=color, label=f"{label}: mean {data.mean():.2f}", ax=ax)
        ax.axvline(data.mean(), color=color, ls="--")
    ax.set(title="Distribution of each group (t-test)", xlabel=pretty(numeric_col),
           ylabel="Density")
    ax.legend()
    fig.tight_layout()
    return fig


def chi_square_plot(result):
    table = result["table"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    pct = table.div(table.sum(axis=1), axis=0) * 100
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax,
                linewidths=1, linecolor="white")
    ax.set(title="Row percentages (contingency table)", xlabel="Major delay (0 = no, 1 = yes)",
           ylabel="")
    fig.tight_layout()
    return fig


def qq_plot(result, column):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    stats.probplot(result["sample"], dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=3, color=MAIN, alpha=0.5)
    ax.get_lines()[1].set(color=WARN, lw=2)
    ax.set_title(f"Normal Q-Q plot: {pretty(column)}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- probability distributions
def fit_normal(df, column="inflation_rate_pct"):
    """Fit a Normal distribution and compare it with the histogram."""
    data = df[column].dropna()
    mu, sigma = stats.norm.fit(data)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(data, bins=50, stat="density", color="#5B7DB1", edgecolor="white", ax=ax)
    x = np.linspace(data.min(), data.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=WARN, lw=2.5,
            label=f"Normal(mu={mu:.2f}, sigma={sigma:.2f})")
    ax.set(title=f"Normal distribution fit: {pretty(column)}", xlabel=pretty(column),
           ylabel="Density")
    ax.legend()
    fig.tight_layout()
    p_within_1sd = stats.norm.cdf(mu + sigma, mu, sigma) - stats.norm.cdf(mu - sigma, mu, sigma)
    actual = ((data > mu - sigma) & (data < mu + sigma)).mean()
    info = {"mean (mu)": round(mu, 3), "std (sigma)": round(sigma, 3),
            "P(within 1 sigma) theory": f"{p_within_1sd * 100:.1f}%",
            "Share within 1 sigma in data": f"{actual * 100:.1f}%"}
    return fig, info


def fit_poisson(df, column="design_change_count"):
    """Compare observed counts with a Poisson distribution (same mean)."""
    data = df[column].dropna().astype(int)
    lam = data.mean()
    k = np.arange(0, data.max() + 1)
    observed = data.value_counts(normalize=True).reindex(k, fill_value=0)
    expected = stats.poisson.pmf(k, lam)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(k - 0.2, observed.values, width=0.4, color=MAIN, label="Observed")
    ax.bar(k + 0.2, expected, width=0.4, color="#F28C28", label=f"Poisson(lambda={lam:.2f})")
    ax.set(title=f"Poisson distribution fit: {pretty(column)}", xlabel=pretty(column),
           ylabel="Probability")
    ax.legend()
    fig.tight_layout()
    info = {"lambda (mean)": round(lam, 3), "variance in data": round(data.var(), 3),
            "P(0 changes) theory": f"{stats.poisson.pmf(0, lam) * 100:.1f}%",
            "P(0 changes) data": f"{(data == 0).mean() * 100:.1f}%",
            "P(5 or more) theory": f"{stats.poisson.sf(4, lam) * 100:.1f}%",
            "P(5 or more) data": f"{(data >= 5).mean() * 100:.1f}%"}
    return fig, info


def fit_exponential(df, column="delay_months"):
    """Fit an Exponential distribution to delays of delayed projects."""
    data = df.loc[df[column] > 0, column]
    loc, scale = stats.expon.fit(data, floc=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(data, bins=60, stat="density", color=MAIN, edgecolor="white", ax=ax)
    x = np.linspace(0, data.quantile(0.99), 300)
    ax.plot(x, stats.expon.pdf(x, loc, scale), color=WARN, lw=2.5,
            label=f"Exponential(mean={scale:.1f} months)")
    ax.set_xlim(0, data.quantile(0.99))
    ax.set(title="Exponential distribution fit: delay of delayed projects",
           xlabel=pretty(column), ylabel="Density")
    ax.legend()
    fig.tight_layout()
    info = {"mean delay (scale)": round(scale, 2),
            "P(delay > 18 m) theory": f"{stats.expon.sf(18, 0, scale) * 100:.1f}%",
            "P(delay > 18 m) data": f"{(data > 18).mean() * 100:.1f}%",
            "P(delay > 36 m) theory": f"{stats.expon.sf(36, 0, scale) * 100:.1f}%",
            "P(delay > 36 m) data": f"{(data > 36).mean() * 100:.1f}%"}
    return fig, info


def binomial_example(df, n_projects=10):
    """Binomial: chance of k major delays among n random projects."""
    p = df["major_delay"].mean()
    k = np.arange(0, n_projects + 1)
    probs = stats.binom.pmf(k, n_projects, p)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(k, probs, color=[WARN if i >= 4 else MAIN for i in k])
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in probs], fontsize=8, padding=2)
    ax.set(title=f"Binomial(n={n_projects}, p={p:.3f}): major delays among {n_projects} projects",
           xlabel="Number of projects with a major delay", ylabel="Probability")
    fig.tight_layout()
    info = {"p (major-delay rate)": round(p, 4),
            "Expected major delays": round(n_projects * p, 2),
            "P(none delayed)": f"{probs[0] * 100:.1f}%",
            "P(4 or more)": f"{stats.binom.sf(3, n_projects, p) * 100:.1f}%"}
    return fig, info


if __name__ == "__main__":
    from src.data_cleaning import load_clean_data
    print(run_standard_tests(load_clean_data()).to_string())
