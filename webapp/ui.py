"""Shared look and feel: page styling, headers, the pipeline diagram and charts."""
import altair as alt
import pandas as pd
import streamlit as st

TITLE = "Sentiment & Emotion Intelligence"
SUBTITLE = "Machine Learning Based Text Sentiment and Emotion Analysis."

# Chart colours from the project's colour-blind-checked palette, per theme. The
# viewer's theme is not always known on the first run, so the muted bar and
# label colours are mid-tones that stay readable on both light and dark pages.
PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "muted": "#a3a29b", "text": "#898781"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "muted": "#a3a29b", "text": "#898781"},
}

TASK_NAMES = {"sentiment": "Sentiment", "emotion": "Emotion"}
ICONS = {
    "positive": "😊", "negative": "😞", "neutral": "😐",
    "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲",
}

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 4rem; padding-bottom: 3rem;}
.app-hero h1 {font-size: 2.15rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; padding: 0;}
.app-hero p {font-size: 1.05rem; opacity: 0.72; margin: 0.3rem 0 0;}
.pipeline {display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem; margin: 0.4rem 0 0.2rem;}
.pipeline .step {flex: 1 1 140px; border: 1px solid rgba(128, 128, 128, 0.32); border-radius: 10px;
  padding: 0.6rem 0.8rem; background: rgba(128, 128, 128, 0.06); font-weight: 600; font-size: 0.93rem;}
.pipeline .step span {display: block; font-weight: 400; opacity: 0.7; font-size: 0.8rem; margin-top: 0.15rem;}
.pipeline .arrow {align-self: center; opacity: 0.45; font-size: 1.1rem;}
.result-label {font-size: 1.85rem; font-weight: 700; line-height: 1.15; margin: 0.1rem 0 0.2rem;}
.result-meta {opacity: 0.72; font-size: 0.9rem;}
.chips span {display: inline-block; padding: 0.18rem 0.65rem; margin: 0 0.35rem 0.4rem 0;
  border-radius: 999px; border: 1px solid rgba(128, 128, 128, 0.35); font-size: 0.85rem;}
@media (max-width: 640px) {
  .block-container {padding-top: 4rem;}
  .app-hero h1 {font-size: 1.6rem;}
  .pipeline .arrow {display: none;}
  .result-label {font-size: 1.5rem;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def theme() -> dict:
    mode = getattr(st.context.theme, "type", None) or "light"
    return PALETTE.get(mode, PALETTE["light"])


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
    st.write("")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pretty(label: str) -> str:
    return f"{ICONS.get(label, '')} {label.capitalize()}".strip()


def pipeline() -> None:
    steps = [
        ("Text", "your input"),
        ("Preprocessing", "decode, lowercase, keep negations"),
        ("TF-IDF", "word & bigram weights"),
        ("Logistic Regression", "one model per task"),
        ("Predictions", "class + probabilities"),
    ]
    parts = []
    for i, (name, note) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">→</div>')
        parts.append(f'<div class="step">{name}<span>{note}</span></div>')
    st.markdown(f'<div class="pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


def chips(items: list[str]) -> None:
    st.markdown('<div class="chips">' + "".join(f"<span>{i}</span>" for i in items) + "</div>",
                unsafe_allow_html=True)


def probability_chart(probabilities: pd.Series, predicted: str) -> alt.Chart:
    """Horizontal bars of class probabilities; the predicted class is highlighted."""
    colors = theme()
    data = pd.DataFrame({
        "class": [label.capitalize() for label in probabilities.index],
        "probability": probabilities.values,
        "predicted": ["Predicted" if label == predicted else "Other" for label in probabilities.index],
    })
    order = data["class"].tolist()
    base = alt.Chart(data).encode(
        y=alt.Y("class:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelPadding=8)),
        x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, grid=True)),
        tooltip=[alt.Tooltip("class:N", title="Class"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color("predicted:N", legend=None,
                        scale=alt.Scale(domain=["Predicted", "Other"],
                                        range=[colors["series"][0], colors["muted"]])))
    labels = base.mark_text(align="left", dx=5, fontSize=12, color=colors["text"]).encode(text=alt.Text("probability:Q", format=".1%"))
    return (bars + labels).properties(height=44 * len(data) + 10)


def grouped_bar_chart(data: pd.DataFrame, x: str, series: str, value: str, x_order: list[str],
                      series_order: list[str], height: int = 300) -> alt.Chart:
    """Grouped vertical bars for scores between 0 and 1."""
    colors = theme()["series"]
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3, width=alt.RelativeBandSize(0.8)).encode(
        x=alt.X(f"{x}:N", sort=x_order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        xOffset=alt.XOffset(f"{series}:N", sort=series_order),
        y=alt.Y(f"{value}:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color(f"{series}:N", sort=series_order, title=None,
                        scale=alt.Scale(domain=series_order, range=colors[:len(series_order)]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{series}:N"), alt.Tooltip(f"{value}:Q", format=".1%")],
    ).properties(height=height)
