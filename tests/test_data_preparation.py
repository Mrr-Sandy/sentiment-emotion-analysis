import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eda  # noqa: E402
from prepare_data import PROCESSED_DIR, TASKS, clean_train  # noqa: E402

EXPECTED_LABELS = {
    "sentiment": ["negative", "neutral", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
}


@unittest.skipUnless(all((PROCESSED_DIR / t / "train.csv").exists() for t in TASKS),
                     "run src/prepare_data.py first")
class PreparedDataTest(unittest.TestCase):
    def test_labels_match_the_expected_classes(self):
        for task in TASKS:
            self.assertEqual(eda.label_order(eda.load_splits(task)), EXPECTED_LABELS[task])

    def test_validation_and_test_are_unchanged(self):
        import pandas as pd
        for task in TASKS:
            raw = eda.load_splits(task)
            for split in ("validation", "test"):
                processed = pd.read_csv(PROCESSED_DIR / task / f"{split}.csv", keep_default_na=False)
                pd.testing.assert_frame_equal(processed, raw[split], check_dtype=False)

    def test_training_split_has_no_conflicting_labels(self):
        import pandas as pd
        for task in TASKS:
            train = pd.read_csv(PROCESSED_DIR / task / "train.csv", keep_default_na=False)
            self.assertEqual(int(train.groupby("text")["label"].nunique().gt(1).sum()), 0)
            self.assertEqual(int(train.duplicated(subset=["text", "label"]).sum()), 0)


class CleanTrainTest(unittest.TestCase):
    def test_removes_conflicts_and_duplicates_only(self):
        import pandas as pd
        train = pd.DataFrame({
            "text": ["a", "a", "b", "b", "c", " "],
            "label": [0, 1, 2, 2, 0, 1],
            "label_name": ["x", "y", "z", "z", "x", "y"],
        })
        self.assertEqual(clean_train(train)["text"].tolist(), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
