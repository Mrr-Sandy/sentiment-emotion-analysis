"""Home / dashboard page."""
import streamlit as st

from webapp import resources, ui

ui.page_header(ui.TITLE, ui.SUBTITLE)

st.markdown(
    "This app reads a piece of English text and answers two questions about it: **what is the overall "
    "attitude** (sentiment) and **which emotion does it express** (emotion). Both answers come from "
    "classical machine-learning models trained on public, labelled Twitter datasets, so you can see "
    "exactly how a transparent, fast baseline performs on real social-media text."
)
st.page_link("webapp/views/analyzer.py", label="Open the Text Analyzer", icon=":material/arrow_forward:")

st.subheader("Two independent classification tasks")
left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Task A · Sentiment analysis**")
    st.caption("Is the text positive, negative or neutral?")
    ui.chips([ui.pretty(c) for c in ("positive", "negative", "neutral")])
with right, st.container(border=True):
    st.markdown("**Task B · Emotion detection**")
    st.caption("Which of six basic emotions does it express?")
    ui.chips([ui.pretty(c) for c in ("joy", "sadness", "anger", "fear", "love", "surprise")])

st.subheader("Model architecture")
ui.pipeline()
st.caption("The same text goes through two separate pipelines, one per task. Preprocessing is built "
           "into each saved TF-IDF vectorizer, so predictions clean text exactly as training did.")

try:
    counts = resources.class_counts()
    metrics = {task: resources.test_metrics(task) for task in resources.TASKS}
except resources.ResourceError as err:
    st.error(str(err))
    st.stop()

st.subheader("Data and results at a glance")
for task, title in (("sentiment", "Sentiment · TweetEval"), ("emotion", "Emotion · dair-ai/emotion")):
    st.markdown(f"**{title}**")
    totals, m = counts[task].sum(), metrics[task]
    cols = st.columns(4)
    cols[0].metric("Training texts", f"{totals['train']:,}", border=True)
    cols[1].metric("Test texts", f"{totals['test']:,}", border=True)
    cols[2].metric("Test accuracy", ui.pct(m["accuracy"]), border=True)
    cols[3].metric("Test macro F1", ui.pct(m["macro_f1"]), border=True,
                   help="F1-score averaged equally over all classes; the main metric for these "
                        "imbalanced datasets.")

st.subheader("Technology stack")
ui.chips(["Python 3.12", "scikit-learn", "TF-IDF", "Logistic Regression", "pandas", "NumPy", "Joblib",
          "Hugging Face Datasets", "Matplotlib", "Altair", "Streamlit"])
