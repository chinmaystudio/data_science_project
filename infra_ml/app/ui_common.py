"""
ui_common.py
------------
Shared UI helpers used by every page:
  * makes the 'src' package importable
  * the portal style (CSS): flat solid colours, square boxes, no gradients
  * header, navigation bar, breadcrumb and footer of the portal
  * cached loading of data, models and saved results
  * small components: page title, section bar, statistic boxes, notices
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, CLEAN_FILE            # noqa: E402
from src.data_loader import load_raw_data                 # noqa: E402
from src.data_cleaning import load_clean_data             # noqa: E402
from src.model_training import load_models                # noqa: E402

# ---------------------------------------------------------------- colours
NAVY = "#1F3B73"        # main colour (header, menu, headings)
NAVY_DARK = "#14284F"
SAFFRON = "#F28C28"
GREEN = "#2E7D32"
RED = "#B22222"
GREY_BG = "#F2F2F2"
BORDER = "#C8CED8"

CSS = f"""
<style>
header[data-testid="stHeader"] {{display: none;}}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none;}}
.stApp {{background: #FFFFFF;}}
.block-container {{padding-top: 0 !important; padding-bottom: 0; max-width: 1200px;}}
h1, h2, h3, h4 {{color: {NAVY};}}

/* ---------- national-colour strip + utility bar + header ---------- */
.flag-strip {{display: flex; height: 5px; margin: 0 -1rem;}}
.flag-strip span {{flex: 1;}}
.utility-bar {{background: #333333; color: #FFFFFF; font-size: 12px; padding: 5px 14px;
  display: flex; justify-content: space-between; margin: 0 -1rem;}}
.portal-header {{display: flex; align-items: center; gap: 16px; padding: 14px 4px;
  border-bottom: 1px solid {BORDER}; background: #FFFFFF;}}
.portal-logo {{width: 58px; height: 58px; background: {NAVY}; color: #FFFFFF;
  display: flex; align-items: center; justify-content: center; font-weight: bold;
  font-size: 22px; border: 3px solid {SAFFRON};}}
.portal-title {{color: {NAVY}; font-size: 26px; font-weight: bold; line-height: 1.1;}}
.portal-sub {{color: #333333; font-size: 15px; margin-top: 3px;}}
.portal-right {{margin-left: auto; text-align: right; font-size: 12px; color: #555555;
  border-left: 1px solid {BORDER}; padding-left: 14px;}}

/* ---------- navigation bar ---------- */
.st-key-govnav {{background: {NAVY}; padding: 0 6px; gap: 0 !important;
  border-bottom: 3px solid {SAFFRON};}}
.st-key-govnav [data-testid="stPageLink"] {{margin: 0;}}
.st-key-govnav a[data-testid="stPageLink-NavLink"] {{background: transparent !important;
  border-radius: 0 !important; padding: 10px 14px !important; margin: 0 !important;}}
.st-key-govnav a[data-testid="stPageLink-NavLink"]:hover {{background: {NAVY_DARK} !important;}}
.st-key-govnav a[data-testid="stPageLink-NavLink"] * {{color: #FFFFFF !important;
  font-size: 14px !important; font-weight: bold;}}
.nav-active {{background: #FFFFFF; color: {NAVY}; font-weight: bold; font-size: 14px;
  padding: 10px 14px; border-top: 3px solid {SAFFRON}; margin-top: 3px;}}
.breadcrumb {{font-size: 13px; color: #555555; padding: 10px 2px 0 2px;}}
.breadcrumb b {{color: {NAVY};}}

/* ---------- page title, sections, boxes ---------- */
.page-title {{border-bottom: 2px solid {NAVY}; padding: 6px 0 8px 0; margin: 6px 0 14px 0;}}
.page-title h1 {{font-size: 26px; margin: 0; padding: 0; color: {NAVY};}}
.page-title p {{margin: 4px 0 0 0; color: #333333; font-size: 15px;}}
.page-title .src {{font-size: 12px; color: #666666;}}
.section-bar {{background: {NAVY}; color: #FFFFFF; font-weight: bold; font-size: 15px;
  padding: 7px 12px; margin: 18px 0 10px 0;}}
.stat-box {{background: #FFFFFF; border: 1px solid {BORDER}; border-top: 4px solid {NAVY};
  padding: 12px 14px; height: 100%;}}
.stat-box .label {{color: #444444; font-size: 13px; font-weight: bold;}}
.stat-box .value {{color: {NAVY}; font-size: 26px; font-weight: bold; margin-top: 4px;}}
.stat-box .sub {{color: #666666; font-size: 12px;}}
.notice {{background: #FFF8E5; border: 1px solid #E3C77A; padding: 10px 14px;
  color: #4A3B00; font-size: 14px; margin: 8px 0 14px 0;}}
.notice-good {{background: #EEF6EE; border: 1px solid #9CC79F; padding: 10px 14px;
  color: #1B4D1E; font-size: 14px; margin: 8px 0 14px 0;}}
.link-box {{border: 1px solid {BORDER}; background: {GREY_BG}; padding: 10px 14px;
  margin-bottom: 6px; font-size: 14px;}}
.link-box b {{color: {NAVY};}}
.risk-badge {{display: inline-block; padding: 5px 16px; color: #FFFFFF; font-weight: bold;}}

/* ---------- streamlit widgets ---------- */
div[data-testid="stTabs"] button {{font-weight: bold; border-radius: 0;}}
div[data-testid="stTabs"] button[aria-selected="true"] {{color: {NAVY};}}
.stButton button, .stFormSubmitButton button, [data-testid="stBaseButton-primaryFormSubmit"] {{
  border-radius: 0 !important;}}
[data-testid="stForm"] {{border-radius: 0; border: 1px solid {BORDER};}}
[data-testid="stDataFrame"] {{border: 1px solid {BORDER};}}
[data-testid="stTable"] table {{border-collapse: collapse; width: 100%; font-size: 14px;}}
[data-testid="stTable"] th {{background: #E6ECF5 !important; color: {NAVY} !important;
  font-weight: bold; border: 1px solid {BORDER} !important;}}
[data-testid="stTable"] td {{border: 1px solid {BORDER} !important;}}

/* ---------- footer ---------- */
.portal-footer {{background: {NAVY}; color: #FFFFFF; margin: 40px -1rem 0 -1rem;
  padding: 18px 24px; font-size: 13px; border-top: 4px solid {SAFFRON};}}
.portal-footer b {{color: #FFFFFF;}}
.portal-footer .cols {{display: flex; gap: 40px; flex-wrap: wrap;}}
.portal-footer .bottom {{border-top: 1px solid #3F5A8F; margin-top: 12px; padding-top: 10px;
  color: #D6DEEB; font-size: 12px;}}
</style>
"""


# ---------------------------------------------------------------- portal frame
def portal_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="flag-strip"><span style="background:#FF9933"></span>'
        '<span style="background:#FFFFFF"></span><span style="background:#138808"></span></div>'
        '<div class="utility-bar"><span>Data Science Mini-Project &nbsp;|&nbsp; '
        'Predictive Analytics Case Study</span><span>Dataset: NIRIKSHAK_Projects_200000.xlsx</span></div>'
        '<div class="portal-header"><div class="portal-logo">NK</div>'
        '<div><div class="portal-title">NIRIKSHAK</div>'
        '<div class="portal-sub">Infrastructure Project Delay &amp; Cost Overrun Prediction Portal</div></div>'
        '<div class="portal-right">Academic Project<br>Python &middot; scikit-learn &middot; Streamlit</div></div>',
        unsafe_allow_html=True)


def portal_nav(pages, current):
    """Horizontal menu bar. pages = list of st.Page, current = page now open."""
    with st.container(key="govnav", horizontal=True, gap=None):
        for page in pages:
            if page.title == current.title:
                st.markdown(f'<div class="nav-active">{page.title}</div>', unsafe_allow_html=True)
            else:
                st.page_link(page, label=page.title)
    crumb = "Home" if current.title == "Home" else f"Home &nbsp;&raquo;&nbsp; <b>{current.title}</b>"
    if current.title == "Home":
        crumb = "<b>Home</b>"
    st.markdown(f'<div class="breadcrumb">You are here: {crumb}</div>', unsafe_allow_html=True)


def portal_footer():
    updated = ""
    path = OUTPUT_DIR / "run_summary.json"
    if path.exists():
        import datetime
        updated = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%d-%m-%Y")
    st.markdown(
        '<div class="portal-footer"><div class="cols">'
        '<div><b>About</b><br>Python ML mini-project on 200,000 infrastructure projects.<br>'
        'EDA &middot; Hypothesis testing &middot; Classification &middot; Regression</div>'
        '<div><b>Data</b><br>NIRIKSHAK_Projects_200000.xlsx (synthetic records derived<br>'
        'from NIRIKSHAK infrastructure archive data)</div>'
        f'<div><b>Models last trained</b><br>{updated or "not trained yet"}</div></div>'
        '<div class="bottom">Disclaimer: this is an academic data-science project and not an official '
        'Government website. Predictions are based on a synthetic dataset and are for learning purposes only.'
        '</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- page components
def page_header(title, subtitle, tag=""):
    src = f'<div class="src">Source code: {tag}</div>' if tag else ""
    st.markdown(f'<div class="page-title"><h1>{title}</h1><p>{subtitle}</p>{src}</div>',
                unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-bar">{title}</div>', unsafe_allow_html=True)


def stat_box(label, value, sub=""):
    return (f'<div class="stat-box"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def cards(items):
    """items = list of (label, value, sub) shown as a row of statistic boxes."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(stat_box(label, value, sub), unsafe_allow_html=True)
    st.write("")


def note(text, kind="note"):
    css = "notice-good" if kind == "good" else "notice"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def show(fig):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def require_training():
    if not CLEAN_FILE.exists() or not (OUTPUT_DIR / "run_summary.json").exists():
        st.error("Models are not trained yet. Run  python train.py  in the project "
                 "folder first, then refresh this page.")
        st.stop()


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner="Loading cleaned data ...")
def get_clean_data():
    return load_clean_data()


@st.cache_data(show_spinner="Loading raw data ...")
def get_raw_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Loading trained models ...")
def get_models(task):
    return load_models(task)


@st.cache_data
def get_output(name):
    path = OUTPUT_DIR / name
    if name.endswith(".json"):
        return json.loads(path.read_text())
    return pd.read_csv(path)
