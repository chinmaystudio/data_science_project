"""Page 6 - Predict a new project (uses src/prediction.py)."""
import matplotlib.pyplot as plt
import numpy as np

from ui_common import (page_header, section, cards, note, require_training, show,
                       get_clean_data, get_models, get_output, st, GREEN, RED, NAVY)
from src.prediction import predict_project, risk_level, risk_flags, compare_with_dataset
from src.config import pretty
from src.eda import MAIN, ACCENT, WARN

page_header("Predict a New Project",
            "Enter the project details known at the time of award. The trained models estimate "
            "the risk of a major delay and the expected cost overrun.",
            tag="src/prediction.py")
require_training()
df = get_clean_data()
clf_models, reg_models = get_models("classification"), get_models("regression")
clf_best = get_output("classification_metrics.csv").iloc[1:].sort_values("ROC-AUC", ascending=False)["model"].iloc[0]
reg_best = get_output("regression_metrics.csv").iloc[1:].sort_values("R2", ascending=False)["model"].iloc[0]

section("Model Settings")
c1, c2, c3 = st.columns(3)
clf_name = c1.selectbox("Classifier (major delay)", list(clf_models), index=list(clf_models).index(clf_best))
reg_name = c2.selectbox("Regressor (cost overrun)", list(reg_models), index=list(reg_models).index(reg_best))
threshold = c3.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                      help="Predict 'major delay' when the probability is at least this value.")

complexity_by_sub = df.groupby("subsector")["complexity_index"].median()
opts = lambda c: sorted(df[c].dropna().unique())

with st.form("project"):
    section("Part A: Project Details")
    c1, c2, c3, c4 = st.columns(4)
    sector = c1.selectbox("Sector", opts("sector"))
    subs = sorted(df.loc[df["sector"] == sector, "subsector"].unique())
    subsector = c2.selectbox("Subsector (sets complexity)", subs)
    terrain = c3.selectbox("Terrain", opts("terrain"))
    area = c4.selectbox("Area type", opts("area_type"))
    c1, c2, c3, c4 = st.columns(4)
    cost = c1.number_input("Sanctioned cost (Rs. crore)", 1.0, 120000.0, 250.0, step=10.0)
    duration = c2.slider("Planned duration (months)", 12, 81, 36)
    contract = c3.selectbox("Contract model", opts("contract_model"))
    funding = c4.selectbox("Funding source", opts("funding_source"))

    section("Part B: Contractor and Bidding")
    c1, c2, c3, c4 = st.columns(4)
    experience = c1.slider("Contractor experience (years)", 0, 60, 15)
    completed = c2.slider("Contractor completed projects", 0, 180, 20)
    bidders = c3.slider("Number of bidders", 1, 17, 4)
    bid_ratio = c4.slider("Bid / estimate ratio", 0.65, 1.35, 0.95, 0.01)
    c1, c2, c3, c4 = st.columns(4)
    dpr = c1.slider("DPR quality score", 1.0, 10.0, 6.5, 0.1)
    design = c2.slider("Expected design changes", 0, 11, 2)
    monitoring = c3.selectbox("Monitoring mechanism", opts("monitoring_mechanism"))
    payment = c4.slider("Payment delay to contractor (days)", 0, 400, 50)

    section("Part C: Land, Clearances and External Risks")
    c1, c2, c3, c4 = st.columns(4)
    land_pct = c1.slider("Land acquired at award (%)", 0.0, 100.0, 75.0, 0.5)
    land_ha = c2.number_input("Land required (ha)", 0.0, 50.0, 1.0, 0.1)
    env = c3.selectbox("Environmental clearance", opts("env_clearance"))
    forest = c4.selectbox("Forest clearance", opts("forest_clearance"))
    c1, c2, c3, c4 = st.columns(4)
    rainfall = c1.slider("Annual rainfall (mm)", 200, 3600, 1100, 50)
    inflation = c2.slider("Inflation (%)", 0.0, 12.0, 5.5, 0.1)
    material = c3.slider("Material price index", 0.5, 1.5, 1.0, 0.01)
    flags = c4.multiselect("Other risks", ["Litigation", "Political change", "Utility shifting", "COVID period"])
    submitted = st.form_submit_button("Submit and Predict", width="stretch", type="primary")


def risk_scale(p, colour):
    """Horizontal scale: Low / Medium / High bands with a marker at p."""
    fig, ax = plt.subplots(figsize=(6, 2.2))
    for start, end, col, label in [(0, 0.30, GREEN, "Low"), (0.30, 0.55, "#C88A00", "Medium"),
                                   (0.55, 1.0, RED, "High")]:
        ax.barh(0, end - start, left=start, height=0.5, color=col, edgecolor="white")
        ax.text((start + end) / 2, 0, label, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
    ax.plot([p, p], [-0.42, 0.42], color="black", lw=3)
    ax.text(p, 0.55, f"{p * 100:.1f}%", ha="center", fontsize=12, fontweight="bold", color="black")
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.85)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([]); ax.grid(False)
    for s in ["left", "top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Probability of a major delay (> 18 months)")
    fig.tight_layout()
    return fig


def show_results(project):
    res = predict_project(project, clf_models[clf_name], reg_models[reg_name], threshold)
    level, colour = risk_level(res["major_delay_probability"])

    section("Prediction Result")
    cards([
        ("Major-delay probability", f"{res['major_delay_probability'] * 100:.1f}%",
         f"{clf_name}, threshold {threshold:.2f}"),
        ("Prediction", "MAJOR DELAY" if res["major_delay_predicted"] else "No major delay",
         f"dataset average {df['major_delay'].mean() * 100:.1f}%"),
        ("Predicted cost overrun", f"{res['cost_overrun_pct']:.1f}%", reg_name),
        ("Expected revised cost", f"Rs. {res['revised_cost_inr_cr']:,.0f} Cr",
         f"Rs. {res['extra_cost_inr_cr']:,.0f} Cr above sanctioned cost"),
    ])
    st.markdown(f'<b>Risk level:</b> <span class="risk-badge" style="background:{colour}">{level}</span>',
                unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        show(risk_scale(res["major_delay_probability"], colour))
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(df["cost_overrun_pct"].clip(-15, 100), bins=60, color=MAIN, edgecolor="white")
        ax.axvline(res["cost_overrun_pct"], color=WARN, lw=3,
                   label=f"This project: {res['cost_overrun_pct']:.1f}%")
        ax.axvline(df["cost_overrun_pct"].mean(), color="black", lw=1.5, ls="--",
                   label=f"Dataset mean: {df['cost_overrun_pct'].mean():.1f}%")
        ax.set(title="Predicted overrun compared with all projects", xlabel="Cost overrun (%)",
               ylabel="Projects")
        ax.legend(fontsize=8)
        fig.tight_layout()
        show(fig)

    c1, c2 = st.columns([1, 1.3])
    with c1:
        section("Risk Flags")
        found = risk_flags(project, df)
        if found:
            for f in found:
                st.markdown(f"- {f}")
        else:
            note("No major warning signs in the inputs.", "good")
    with c2:
        section("Comparison with Typical Project")
        comp = compare_with_dataset(project, df)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ratio = (comp["This project"] / comp["Dataset median"].replace(0, np.nan)).fillna(1)
        ax.barh([pretty(i) for i in comp.index], ratio,
                color=[WARN if (r > 1.15 or r < 0.85) else ACCENT for r in ratio])
        ax.axvline(1, color="black", lw=1)
        ax.set(title="This project / dataset median (1 = typical)", xlabel="Ratio")
        fig.tight_layout()
        show(fig)

    section("What-if Analysis: Land Acquired at Award")
    xs = np.arange(30, 101, 5)
    probs = [predict_project(dict(project, land_acquired_pct_at_award=float(x)),
                             clf_models[clf_name], reg_models[reg_name])["major_delay_probability"]
             for x in xs]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(xs, np.array(probs) * 100, marker="o", color=MAIN, lw=2.2)
    ax.axvline(land_pct, color=WARN, ls="--", label="Current input")
    ax.set(xlabel="Land acquired at award (%)", ylabel="Major-delay probability (%)",
           title="Same project with different land acquisition")
    ax.legend()
    fig.tight_layout()
    show(fig)
    note("Predictions show statistical patterns learned from a <b>synthetic</b> dataset. Use them "
         "as an early-warning signal, not as a guarantee.")


if submitted:
    show_results({
        "sector": sector, "area_type": area, "terrain": terrain, "contract_model": contract,
        "funding_source": funding, "monitoring_mechanism": monitoring, "env_clearance": env,
        "forest_clearance": forest, "sanctioned_cost_inr_cr": cost,
        "planned_duration_months": duration, "complexity_index": float(complexity_by_sub[subsector]),
        "num_bidders": bidders, "bid_to_estimate_ratio": bid_ratio,
        "contractor_experience_yrs": experience, "contractor_completed_projects": completed,
        "land_required_ha": land_ha, "land_acquired_pct_at_award": land_pct,
        "utility_shifting_required": int("Utility shifting" in flags), "design_change_count": design,
        "dpr_quality_score": dpr, "avg_annual_rainfall_mm": rainfall,
        "political_change_during_exec": int("Political change" in flags),
        "litigation_flag": int("Litigation" in flags), "covid_affected": int("COVID period" in flags),
        "avg_payment_delay_days": payment, "inflation_rate_pct": inflation,
        "material_price_index": material,
    })
else:
    st.info("Fill in Parts A, B and C and press Submit and Predict.")
