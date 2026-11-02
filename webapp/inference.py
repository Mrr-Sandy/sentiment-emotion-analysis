"""Input validation and prediction for the web app (no Streamlit code).

Both classifiers come from src/predict.py's load_classifier(); their saved
vectorizers apply the training preprocessing, so nothing is re-implemented here.
"""
import re
from dataclasses import dataclass

import pandas as pd

MAX_CHARS = 2000
LETTER_RE = re.compile(r"[^\W\d_]")  # any letter, in any script

# Example inputs shown in the analyzer, keyed by the button label.
EXAMPLES = {
    "Positive": "I absolutely love this phone, best purchase ever!",
    "Negative": "Worst customer service ever. Never again.",
    "Neutral": "The train leaves at 9 tomorrow morning.",
    "Joy": "i feel so grateful and happy today",
    "Sadness": "i feel really lonely tonight",
    "Anger": "i am so annoyed with my noisy neighbours",
    "Fear": "i feel nervous about my exam results",
    "Love": "i feel so loved by my friends",
    "Surprise": "i was shocked to see him there",
}


class InvalidInput(ValueError):
    """Raised with a message that can be shown to the user as-is."""


@dataclass
class TaskPrediction:
    label: str
    confidence: float
    probabilities: pd.Series  # one value per class, in label order
    known_terms: int  # TF-IDF features found in the text; 0 means the model saw nothing it knows


def validate(text: str | None) -> str:
    """Return the stripped text, or raise InvalidInput with a user-facing message."""
    if text is None or not text.strip():
        raise InvalidInput("Please enter some text to analyse.")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise InvalidInput(f"The text is {len(text):,} characters long. Please shorten it to "
                           f"{MAX_CHARS:,} characters or fewer.")
    if not LETTER_RE.search(text):
        raise InvalidInput("The text contains no words to analyse. Please enter a sentence in English.")
    return text


def predict_task(classifier, text: str) -> TaskPrediction:
    probabilities = classifier.predict_proba([text]).iloc[0]
    label = classifier.predict([text])[0]
    return TaskPrediction(
        label=label,
        confidence=float(probabilities[label]),
        probabilities=probabilities,
        known_terms=int(classifier.vectorizer.transform([text]).nnz),
    )


def analyze(text: str | None, classifiers: dict) -> dict[str, TaskPrediction]:
    """Validate the text and run it independently through every classifier."""
    text = validate(text)
    return {task: predict_task(classifier, text) for task, classifier in classifiers.items()}


def preprocessed(text: str, classifier) -> str:
    """The text exactly as the classifier's vectorizer sees it after preprocessing."""
    return classifier.vectorizer.build_preprocessor()(text)
