"""Text Analyzer page: run both classifiers on the user's text."""
import logging

import streamlit as st

from webapp import inference, resources, ui

log = logging.getLogger(__name__)

ui.page_header("Text Analyzer", "Enter an English sentence to get its sentiment and emotion.")

try:
    classifiers = resources.get_classifiers()
except resources.ResourceError as err:
    st.error(str(err))
    st.stop()


def use_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["text"] = inference.EXAMPLES[choice]
        st.session_state.pop("result", None)


st.pills("Try an example", list(inference.EXAMPLES), key="example", on_change=use_example)
text = st.text_area("Your text", key="text", height=140, max_chars=inference.MAX_CHARS,
                    placeholder="e.g. I can't believe how well the exam went, I'm so relieved!")

if st.button("Analyze", type="primary", icon=":material/auto_awesome:"):
    try:
        st.session_state["result"] = (text, inference.analyze(text, classifiers))
    except inference.InvalidInput as err:
        st.session_state.pop("result", None)
        st.warning(str(err), icon=":material/info:")
    except Exception:  # noqa: BLE001 - logged for the developer, generic message for the user
        log.exception("Prediction failed")
        st.session_state.pop("result", None)
        st.error("Something went wrong while analysing the text. Please try again.")

result = st.session_state.get("result")
if result and result[0] == text:
    analysed_text, predictions = result
    sentiment, emotion = predictions["sentiment"], predictions["emotion"]

    st.subheader("Result")
    st.success(f"This text reads as **{sentiment.label.capitalize()}**, and its dominant emotion is "
               f"**{emotion.label.capitalize()}**.")

    for task, prediction in predictions.items():
        if prediction.known_terms == 0:
            st.warning(f"None of the words in this text appear in the {ui.TASK_NAMES[task].lower()} "
                       "model's vocabulary, so its prediction only reflects the class balance it was "
                       "trained on.", icon=":material/warning:")

    columns = st.columns(2)
    for column, (task, prediction) in zip(columns, predictions.items()):
        with column, st.container(border=True):
            st.caption(ui.TASK_NAMES[task].upper())
            st.markdown(f'<div class="result-label">{ui.pretty(prediction.label)}</div>'
                        f'<div class="result-meta">Model confidence {ui.pct(prediction.confidence)}</div>',
                        unsafe_allow_html=True)
            st.altair_chart(ui.probability_chart(prediction.probabilities, prediction.label),
                            width="stretch")

    st.caption("Confidence is the model's estimated probability for its top class. It is not a guarantee "
               "that the prediction is correct: on unseen test data the sentiment model reaches about "
               "60% macro F1 and the emotion model about 84%.")
    with st.expander("What the models saw"):
        st.markdown("After preprocessing (identical to training), the text becomes:")
        st.code(inference.preprocessed(analysed_text, classifiers["sentiment"]), language=None)
        st.markdown(f"Known TF-IDF features found: sentiment **{sentiment.known_terms}**, "
                    f"emotion **{emotion.known_terms}**.")
