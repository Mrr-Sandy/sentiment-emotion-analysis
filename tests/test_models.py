import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import preprocessing  # noqa: E402
from modeling import MODELS_DIR  # noqa: E402
from predict import load_classifier  # noqa: E402

SAMPLES = {
    "sentiment": {
        "I absolutely love this, best day ever!": "positive",
        "This is the worst service I have ever had, totally disappointed.": "negative",
        "The meeting is scheduled for Monday at 10am.": None,
        "I don't think this is good at all": None,
    },
    "emotion": {
        "i feel so happy and excited about the new job": "joy",
        "i feel lonely and miserable since she left": "sadness",
        "i am furious and so angry at them": "anger",
        "i feel scared and nervous about tomorrow": "fear",
        "i feel so loved and caring towards my family": "love",
        "i was amazed and shocked by the news": "surprise",
    },
}
EXPECTED_LABELS = {
    "sentiment": {"negative", "neutral", "positive"},
    "emotion": {"sadness", "joy", "love", "anger", "fear", "surprise"},
}


@unittest.skipUnless(all((MODELS_DIR / t / "model.joblib").exists() for t in SAMPLES),
                     "train the models first")
class SavedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifiers = {task: load_classifier(task) for task in SAMPLES}

    def test_predictions_belong_to_the_expected_classes(self):
        for task, classifier in self.classifiers.items():
            predictions = classifier.predict(list(SAMPLES[task]))
            self.assertEqual(set(classifier.labels), EXPECTED_LABELS[task])
            self.assertTrue(set(predictions) <= EXPECTED_LABELS[task], msg=f"{task}: {predictions}")

    def test_clear_examples_get_the_obvious_class(self):
        for task, classifier in self.classifiers.items():
            for text, expected in SAMPLES[task].items():
                if expected:
                    self.assertEqual(classifier.predict(text), [expected], msg=f"{task}: {text!r}")

    def test_probabilities_are_valid(self):
        for task, classifier in self.classifiers.items():
            probabilities = classifier.predict_proba(list(SAMPLES[task]))
            self.assertEqual(list(probabilities.columns), classifier.labels)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)

    def test_inference_uses_the_training_preprocessing(self):
        for task, classifier in self.classifiers.items():
            vectorizer = classifier.vectorizer
            self.assertIs(vectorizer.preprocessor, preprocessing.clean_text)
            self.assertEqual(vectorizer.token_pattern, preprocessing.TOKEN_PATTERN)
            self.assertIsNone(vectorizer.stop_words)
            texts = list(SAMPLES[task])
            # Cleaning by hand and vectorizing the result gives exactly the same features.
            manual = vectorizer.transform([preprocessing.clean_text(t) for t in texts])
            self.assertEqual((classifier._features(texts) != manual).nnz, 0)

    def test_metadata_records_the_selected_configuration(self):
        for task, classifier in self.classifiers.items():
            meta = classifier.metadata
            self.assertEqual(meta["task"], task)
            self.assertEqual(meta["preprocessing"]["version"], preprocessing.PREPROCESSING_VERSION)
            self.assertEqual(meta["tfidf"]["n_features"], len(classifier.vectorizer.vocabulary_))
            self.assertEqual(sorted(meta["model_classes"]), sorted(meta["labels"]))

    def test_missing_model_gives_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_classifier("sentiment", models_dir=MODELS_DIR / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
