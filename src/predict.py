"""Load the saved models and classify new text.

    from predict import load_classifier
    sentiment = load_classifier("sentiment")
    sentiment.predict(["I can't wait for the weekend!"])   # -> ["positive"]

The saved vectorizer carries preprocessing.clean_text as its preprocessor, so
predictions apply exactly the cleaning used in training. src/ must be on
sys.path when loading (the scripts and tests in this project arrange that).
"""
import json
from pathlib import Path

import joblib
import pandas as pd

import preprocessing
from modeling import MODELS_DIR

TRAIN_SCRIPTS = {"sentiment": "src/train_sentiment.py", "emotion": "src/train_emotion.py"}


class TextClassifier:
    """A fitted TF-IDF vectorizer and Logistic Regression model for one task."""

    def __init__(self, task: str, vectorizer, model, metadata: dict):
        self.task = task
        self.vectorizer = vectorizer
        self.model = model
        self.metadata = metadata
        self.labels = metadata["labels"]

    def _features(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(list(texts))

    def predict(self, texts) -> list[str]:
        """Predicted class name for each text."""
        return self.model.predict(self._features(texts)).tolist()

    def predict_proba(self, texts) -> pd.DataFrame:
        """Class probabilities, one row per text and one column per class (label order)."""
        probabilities = pd.DataFrame(self.model.predict_proba(self._features(texts)),
                                     columns=self.model.classes_)
        return probabilities[self.labels]


def load_classifier(task: str, models_dir: Path = MODELS_DIR) -> TextClassifier:
    """Load models/<task>/ and check it matches the current preprocessing code."""
    folder = Path(models_dir) / task
    paths = [folder / name for name in ("vectorizer.joblib", "model.joblib", "metadata.json")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}. Run: python {TRAIN_SCRIPTS.get(task, '...')}")

    vectorizer, model = joblib.load(paths[0]), joblib.load(paths[1])
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    if metadata["preprocessing"]["version"] != preprocessing.PREPROCESSING_VERSION:
        raise RuntimeError(
            f"{task} model was trained with preprocessing version {metadata['preprocessing']['version']}, "
            f"but the code is at version {preprocessing.PREPROCESSING_VERSION}. Retrain the model."
        )
    if vectorizer.preprocessor is not preprocessing.clean_text:
        raise RuntimeError(f"{task} vectorizer does not use preprocessing.clean_text")
    return TextClassifier(task, vectorizer, model, metadata)
