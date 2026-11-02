"""Sentiment & Emotion Intelligence: Streamlit web app.

    streamlit run app.py

Uses the models saved in models/ by Phase 2; nothing is trained at startup.
"""
import logging

import streamlit as st

from webapp import ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title=ui.TITLE, page_icon="🧠", layout="wide")
ui.inject_css()

pages = [
    st.Page("webapp/views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("webapp/views/analyzer.py", title="Text Analyzer", icon=":material/psychology:"),
    st.Page("webapp/views/performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("webapp/views/dataset.py", title="Dataset Analysis", icon=":material/database:"),
]

with st.sidebar:
    st.markdown(f"### 🧠 {ui.TITLE}")
    st.caption(ui.SUBTITLE)

navigation = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("TF-IDF + Logistic Regression models trained on TweetEval (sentiment) and "
               "dair-ai/emotion. Switch light/dark theme under ⋮ → Settings.")

navigation.run()
