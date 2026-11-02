import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modeling import MODELS_DIR  # noqa: E402
from predict import load_classifier  # noqa: E402
from webapp import inference  # noqa: E402

SENTIMENTS = {"Positive", "Negative", "Neutral"}
EMOTIONS = {"Joy", "Sadness", "Anger", "Fear", "Love", "Surprise"}


class ValidationTest(unittest.TestCase):
    def test_rejects_empty_and_whitespace(self):
        for text in [None, "", "   \n\t "]:
            with self.assertRaisesRegex(inference.InvalidInput, "enter some text"):
                inference.validate(text)

    def test_rejects_text_without_words(self):
        for text in ["12345", "!!! ???", "3.14 + 2"]:
            with self.assertRaisesRegex(inference.InvalidInput, "no words"):
                inference.validate(text)

    def test_rejects_overlong_text(self):
        with self.assertRaisesRegex(inference.InvalidInput, "shorten"):
            inference.validate("word " * 500)

    def test_strips_valid_text(self):
        self.assertEqual(inference.validate("  I feel great  "), "I feel great")


@unittest.skipUnless((MODELS_DIR / "sentiment" / "model.joblib").exists(), "train the models first")
class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifiers = {task: load_classifier(task) for task in ("sentiment", "emotion")}

    def test_examples_cover_every_class(self):
        self.assertEqual(set(inference.EXAMPLES), SENTIMENTS | EMOTIONS)
        for name, text in inference.EXAMPLES.items():
            result = inference.analyze(text, self.classifiers)
            task = "sentiment" if name in SENTIMENTS else "emotion"
            self.assertEqual(result[task].label, name.lower(), msg=text)

    def test_both_models_return_valid_distributions(self):
        result = inference.analyze("I can't believe it, this is amazing!", self.classifiers)
        self.assertEqual(set(result), {"sentiment", "emotion"})
        for task, prediction in result.items():
            self.assertAlmostEqual(float(prediction.probabilities.sum()), 1.0, places=6)
            self.assertEqual(prediction.confidence, float(prediction.probabilities.max()))
            self.assertEqual(list(prediction.probabilities.index), self.classifiers[task].labels)
            self.assertGreater(prediction.known_terms, 0)

    def test_unknown_words_are_flagged(self):
        result = inference.analyze("qwxzv blorptang", self.classifiers)
        self.assertTrue(all(p.known_terms == 0 for p in result.values()))

    def test_preprocessed_text_matches_training_preprocessing(self):
        import preprocessing
        text = "I don't LOVE this &amp; that!"
        self.assertEqual(inference.preprocessed(text, self.classifiers["sentiment"]),
                         preprocessing.clean_text(text))


if __name__ == "__main__":
    unittest.main()
