"""Model Performance page: Phase 2 test results from reports/metrics/."""
import pandas as pd
import streamlit as st

from webapp import resources, ui

ui.page_header("Model Performance", "Results of the saved models on the untouched test splits (Phase 2).")

try:
    metrics = {task: resources.test_metrics(task) for task in resources.TASKS}
except resources.ResourceError as err:
    st.error(str(err))
    st.stop()

SUMMARY = {"Accuracy": "accuracy", "Macro precision": "macro_precision", "Macro recall": "macro_recall",
           "Macro F1": "macro_f1", "Weighted F1": "weighted_f1"}
NAMES = {"sentiment": "Sentiment (TweetEval)", "emotion": "Emotion (dair-ai)"}

for task in resources.TASKS:
    st.markdown(f"**{NAMES[task]}** · {metrics[task]['test_rows']:,} test texts")
    for column, (label, key) in zip(st.columns(len(SUMMARY)), SUMMARY.items()):
        column.metric(label, ui.pct(metrics[task][key]), border=True)

st.subheader("Comparison")
comparison = pd.DataFrame([
    {"metric": label, "model": NAMES[task], "score": metrics[task][key]}
    for task in resources.TASKS for label, key in SUMMARY.items()
])
st.altair_chart(ui.grouped_bar_chart(comparison, "metric", "model", "score", list(SUMMARY),
                                     list(NAMES.values())), width="stretch")

with st.expander("What the metrics mean", expanded=True):
    st.markdown(
        "- **Accuracy:** the share of test texts classified correctly.\n"
        "- **Precision** (per class): of the texts the model put in a class, the share that truly belong to it.\n"
        "- **Recall** (per class): of the texts that truly belong to a class, the share the model found.\n"
        "- **F1-score:** the harmonic mean of precision and recall; high only when both are high.\n"
        "- **Macro** scores average the per-class scores with equal weight for every class; **weighted** "
        "scores weight each class by its number of test texts.\n\n"
        "**Why macro F1?** Both datasets are imbalanced: in the emotion data *joy* is 9 times more common "
        "than *surprise*. A model can reach high accuracy by doing well on the large classes while failing "
        "the small ones. Macro F1 gives every class the same say, so it exposes that failure, which is why "
        "the models were selected on validation macro F1."
    )

for tab, task in zip(st.tabs([NAMES[t] for t in resources.TASKS]), resources.TASKS):
    with tab:
        m = metrics[task]
        model = m["model"]
        st.markdown(
            f"**Selected model:** TF-IDF `{model['tfidf']['name']}` ({model['tfidf']['n_features']:,} features) "
            f"+ Logistic Regression (C = {model['logistic_regression']['C']}, class weights "
            f"`{model['logistic_regression']['class_weight']}`). Validation macro F1 "
            f"{ui.pct(model['validation']['macro_f1'])}; test macro F1 {ui.pct(m['macro_f1'])}."
        )
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown("**Confusion matrix**")
            try:
                st.image(str(resources.figure(f"{task}_confusion_matrix")), width="stretch")
            except resources.ResourceError as err:
                st.warning(str(err))
        with right:
            st.markdown("**Per-class precision, recall and F1**")
            labels = list(m["per_class"])
            per_class = pd.DataFrame([
                {"class": label.capitalize(), "score_type": name, "score": scores[key]}
                for label, scores in m["per_class"].items()
                for name, key in (("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1"))
            ])
            st.altair_chart(ui.grouped_bar_chart(per_class, "class", "score_type", "score",
                                                 [label.capitalize() for label in labels],
                                                 ["Precision", "Recall", "F1-score"], height=280),
                            width="stretch")
            table = pd.DataFrame(m["per_class"]).T.rename(columns={
                "precision": "Precision", "recall": "Recall", "f1": "F1-score", "support": "Test texts"})
            table.index = [label.capitalize() for label in table.index]
            st.dataframe(
                table.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1-score": "{:.1%}",
                                    "Test texts": "{:,.0f}"}),
                width="stretch",
            )
