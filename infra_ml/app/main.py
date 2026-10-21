"""
main.py  -  entry point of the UI.

Run from the project folder:
    streamlit run app/main.py

This file builds the portal frame (header, menu bar, breadcrumb, footer)
and opens the selected page from the 'views' folder. Each page is a
separate file so that every step of the project is easy to find.
"""
import streamlit as st

st.set_page_config(page_title="NIRIKSHAK Prediction Portal", page_icon=":material/analytics:",
                   layout="wide", initial_sidebar_state="collapsed")

from ui_common import portal_header, portal_nav, portal_footer   # noqa: E402

PAGES = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/data_overview.py", title="Data Overview", url_path="data-overview"),
    st.Page("views/cleaning_preprocessing.py", title="Cleaning & Preprocessing", url_path="cleaning"),
    st.Page("views/eda.py", title="EDA", url_path="eda"),
    st.Page("views/hypothesis_testing.py", title="Hypothesis Testing", url_path="hypothesis-testing"),
    st.Page("views/model_performance.py", title="Model Performance", url_path="model-performance"),
    st.Page("views/predict.py", title="Predict", url_path="predict"),
]

current = st.navigation(PAGES, position="hidden")   # menu is drawn by portal_nav
portal_header()
portal_nav(PAGES, current)
current.run()
portal_footer()
