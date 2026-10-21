"""
eda.py
------
STEP 4 - Exploratory Data Analysis (EDA) with Matplotlib and Seaborn.

Every function returns a Matplotlib Figure, so the same chart can be shown
in the Streamlit UI or saved as a PNG file.

Techniques covered:
  * Summary statistics         -> summary_statistics(), category_summary()
  * Data distribution analysis -> histograms, KDE, boxplots, violin plots
  * Correlation analysis       -> heatmap, scatter with trend line
  * Group comparison           -> major-delay rate / overrun by category
"""
import matplotlib
matplotlib.use("Agg")                       # draw without a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import NUMERIC_FEATURES, pretty

# ---------------------------------------------------------------- style
# Flat, solid colours in the style of a government portal
PALETTE = ["#1F3B73", "#F28C28", "#2E7D32", "#B22222", "#5B7DB1",
           "#6D6D6D", "#B8860B", "#4E7A68"]
MAIN = "#1F3B73"      # navy
ACCENT = "#2E7D32"    # green
WARN = "#B22222"      # red


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
    })


set_style()


def _fig(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ---------------------------------------------------------------- tables
def summary_statistics(df, columns=None):
    """Mean, median, std, min, quartiles, max, skewness for numeric columns."""
    columns = columns or NUMERIC_FEATURES + ["delay_months", "cost_overrun_pct"]
    stats = df[columns].describe().T
    stats["median"] = df[columns].median()
    stats["skewness"] = df[columns].skew()
    stats["missing"] = df[columns].isna().sum()
    stats.index = [pretty(c) for c in stats.index]
    return stats[["count", "mean", "median", "std", "min", "25%", "75%",
                  "max", "skewness", "missing"]].round(2)


def category_summary(df, column):
    """Projects, major-delay rate and mean overrun for each category."""
    table = df.groupby(column).agg(
        projects=("project_id", "count"),
        major_delay_rate_pct=("major_delay", lambda s: round(s.mean() * 100, 2)),
        mean_delay_months=("delay_months", "mean"),
        mean_cost_overrun_pct=("cost_overrun_pct", "mean"),
    ).round(2)
    return table.sort_values("major_delay_rate_pct", ascending=False)


def outlier_table(df, columns=None):
    """Number of outliers per column using the 1.5 x IQR rule."""
    columns = columns or NUMERIC_FEATURES + ["delay_months", "cost_overrun_pct"]
    rows = []
    for c in columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
        rows.append({"variable": pretty(c), "outliers": int(n_out),
                     "outlier_pct": round(n_out / len(df) * 100, 2)})
    return pd.DataFrame(rows).sort_values("outliers", ascending=False)


# ---------------------------------------------------------------- distributions
def histogram(df, column, bins=50, kde=True, clip_quantile=0.995):
    """Histogram + KDE curve of one numeric column (extreme tail clipped)."""
    fig, ax = _fig()
    data = df[column].dropna()
    data = data[data <= data.quantile(clip_quantile)]
    sns.histplot(data, bins=bins, kde=kde, color=MAIN, edgecolor="white",
                 alpha=1, ax=ax, line_kws=dict(color="#F28C28", lw=2))
    ax.axvline(data.mean(), color=WARN, ls="--", lw=1.6,
               label=f"Mean = {data.mean():.2f}")
    ax.axvline(data.median(), color=ACCENT, ls="-", lw=1.6,
               label=f"Median = {data.median():.2f}")
    ax.set(title=f"Distribution of {pretty(column)}", xlabel=pretty(column),
           ylabel="Number of projects")
    ax.legend()
    fig.tight_layout()
    return fig


def histogram_grid(df, columns):
    """Small histograms of several numeric columns side by side."""
    n = len(columns)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    for ax, c in zip(np.ravel(axes), columns):
        sns.histplot(df[c].dropna(), bins=40, color=PALETTE[columns.index(c) % 8],
                     edgecolor="white", ax=ax)
        ax.set(title=pretty(c), xlabel="", ylabel="")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def target_overview(df):
    """Donut of major-delay share + histogram of cost overrun."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    counts = df["major_delay"].value_counts().sort_index()
    a1.pie(counts, labels=["No major delay", "Major delay (>18 m)"],
           autopct="%1.1f%%", startangle=90, colors=[MAIN, WARN],
           wedgeprops=dict(edgecolor="white"),
           textprops=dict(fontsize=10))
    a1.set_title("Classification target")
    data = df["cost_overrun_pct"].clip(upper=100)
    sns.histplot(data, bins=60, color=MAIN, edgecolor="white", ax=a2)
    a2.set(title="Regression target", xlabel="Cost overrun (%) - clipped at 100",
           ylabel="Projects")
    fig.tight_layout()
    return fig


def boxplot_by_category(df, numeric_col, cat_col, ylim=None):
    """Boxplot of a numeric column for every category."""
    fig, ax = _fig(9, 4.6)
    order = df.groupby(cat_col)[numeric_col].median().sort_values().index
    sns.boxplot(data=df, x=cat_col, y=numeric_col, order=order, hue=cat_col,
                palette=PALETTE[:df[cat_col].nunique()] if df[cat_col].nunique() <= 8 else None, legend=False, fliersize=1, ax=ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set(title=f"{pretty(numeric_col)} by {pretty(cat_col)}",
           xlabel=pretty(cat_col), ylabel=pretty(numeric_col))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def violin_by_category(df, numeric_col, cat_col, sample_size=20000):
    """Violin plot: shape of the distribution in each category."""
    fig, ax = _fig(9, 4.6)
    sample = df.sample(min(sample_size, len(df)), random_state=1)
    sns.violinplot(data=sample, x=cat_col, y=numeric_col, hue=cat_col,
                   palette=PALETTE[:sample[cat_col].nunique()] if sample[cat_col].nunique() <= 8 else None, legend=False, cut=0, inner="quartile", ax=ax)
    ax.set(title=f"{pretty(numeric_col)} distribution by {pretty(cat_col)}",
           xlabel=pretty(cat_col), ylabel=pretty(numeric_col))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- group comparison
def rate_by_category(df, cat_col, target="major_delay"):
    """Bar chart of the % of projects with the target = 1 in each category."""
    fig, ax = _fig(9, 4.6)
    rates = (df.groupby(cat_col)[target].mean() * 100).sort_values(ascending=False)
    bars = ax.bar(rates.index.astype(str), rates.values,
                  color=[PALETTE[i % 8] for i in range(len(rates))])
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(df[target].mean() * 100, color="grey", ls="--", lw=1,
               label=f"Overall {df[target].mean() * 100:.1f}%")
    ax.set(title=f"{pretty(target)} rate by {pretty(cat_col)}",
           xlabel=pretty(cat_col), ylabel="Projects (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    return fig


def binned_rate(df, numeric_col, target="major_delay", bins=8):
    """Major-delay rate across equal-size groups (quantiles) of a number."""
    fig, ax = _fig(9, 4.4)
    groups = pd.qcut(df[numeric_col], q=bins, duplicates="drop")
    rates = df.groupby(groups, observed=True)[target].mean() * 100
    ax.plot(range(len(rates)), rates.values, marker="o", color=MAIN, lw=2.2)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{i.left:.0f}-{i.right:.0f}" if i.right > 10 else
                        f"{i.left:.2f}-{i.right:.2f}" for i in rates.index],
                       rotation=30, fontsize=8)
    ax.set(title=f"{pretty(target)} rate across {pretty(numeric_col)} groups",
           xlabel=pretty(numeric_col) + " (equal-size groups)",
           ylabel="Projects (%)")
    fig.tight_layout()
    return fig


def stacked_delay_mix(df, cat_col):
    """100 % stacked bar of delay category within each group."""
    order = ["On time", "Minor (<=6m)", "Moderate (6-18m)", "Major (18-36m)",
             "Severe (>36m)"]
    table = pd.crosstab(df[cat_col], df["delay_category"], normalize="index")[order] * 100
    fig, ax = _fig(9, 4.6)
    table.plot(kind="barh", stacked=True, ax=ax,
               color=["#2E7D32", "#8BAA3C", "#F2A900", "#E06C00", "#B22222"],
               width=0.75, edgecolor="white")
    ax.set(title=f"Delay category mix by {pretty(cat_col)}",
           xlabel="Share of projects (%)", ylabel="")
    ax.legend(title="Delay", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def projects_per_year(df):
    """Line chart: number of projects awarded each year."""
    fig, ax = _fig(9, 4)
    counts = df["award_year"].value_counts().sort_index()
    ax.plot(counts.index, counts.values, marker="o", color=MAIN, lw=2)
    ax.set(title="Projects awarded per year", xlabel="Award year",
           ylabel="Projects")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- correlation
def correlation_heatmap(df, columns=None):
    columns = columns or NUMERIC_FEATURES + ["delay_months", "cost_overrun_pct"]
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.5,
                xticklabels=[pretty(c) for c in columns],
                yticklabels=[pretty(c) for c in columns], ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title("Correlation matrix (lower triangle)")
    fig.tight_layout()
    return fig


def correlation_with_target(df, target="cost_overrun_pct"):
    """Horizontal bar of each feature's correlation with the target."""
    corr = df[NUMERIC_FEATURES].corrwith(df[target]).sort_values()
    fig, ax = _fig(8, 6)
    colors = [WARN if v > 0 else MAIN for v in corr.values]
    ax.barh([pretty(c) for c in corr.index], corr.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=f"Correlation of each feature with {pretty(target)}",
           xlabel="Pearson r")
    fig.tight_layout()
    return fig


def scatter_with_trend(df, x, y, hue=None, sample_size=4000):
    """Scatter plot of a random sample with a fitted straight line."""
    fig, ax = _fig(8.5, 4.8)
    s = df.sample(min(sample_size, len(df)), random_state=7)
    if hue:
        sns.scatterplot(data=s, x=x, y=y, hue=hue, alpha=0.45, s=14,
                        palette=PALETTE[:s[hue].nunique()], ax=ax)
    else:
        sns.scatterplot(data=s, x=x, y=y, alpha=0.35, s=14, color=MAIN, ax=ax)
    sns.regplot(data=s, x=x, y=y, scatter=False, color=WARN, ax=ax,
                line_kws=dict(lw=2))
    r = df[[x, y]].corr().iloc[0, 1]
    ax.set(title=f"{pretty(y)} vs {pretty(x)}  (r = {r:.3f})",
           xlabel=pretty(x), ylabel=pretty(y))
    fig.tight_layout()
    return fig


def pair_plot(df, columns, hue="major_delay", sample_size=1500):
    """Seaborn pair plot (scatter matrix) on a small random sample."""
    s = df.sample(sample_size, random_state=3)[columns + [hue]]
    s = s.rename(columns={c: pretty(c) for c in columns + [hue]})
    grid = sns.pairplot(s, hue=pretty(hue), palette=[MAIN, WARN],
                        plot_kws=dict(alpha=0.45, s=12), corner=True, height=2)
    return grid.figure
