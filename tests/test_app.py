import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGES = {
    "home": "webapp/views/home.py",
    "analyzer": "webapp/views/analyzer.py",
    "performance": "webapp/views/performance.py",
    "dataset": "webapp/views/dataset.py",
}


@unittest.skipUnless((ROOT / "models" / "sentiment" / "model.joblib").exists(), "train the models first")
class AppPagesTest(unittest.TestCase):
    def open(self, page: str) -> AppTest:
        app = AppTest.from_file(APP, default_timeout=180)
        app.run()
        if page != "home":
            app.switch_page(PAGES[page]).run()
        self.assertFalse(app.exception, msg=[e.value for e in app.exception])
        self.assertFalse(app.error, msg=[e.value for e in app.error])
        return app

    def test_home_shows_real_results(self):
        app = self.open("home")
        values = {m.label: m.value for m in app.metric}
        self.assertEqual(values["Test macro F1"], "84.0%")  # last task shown is emotion
        self.assertIn("59.4%", [m.value for m in app.metric])

    def test_performance_matches_the_evaluation_report(self):
        app = self.open("performance")
        values = [m.value for m in app.metric]
        for expected in ["59.8%", "58.8%", "60.2%", "59.4%", "88.4%", "81.9%", "87.1%", "84.0%", "88.7%"]:
            self.assertIn(expected, values)

    def test_dataset_page_shows_split_sizes(self):
        app = self.open("dataset")
        values = [m.value for m in app.metric]
        for expected in ["45,615", "12,284", "16,000", "2,000"]:
            self.assertIn(expected, values)

    def test_analyzer_predicts_both_tasks(self):
        app = self.open("analyzer")
        app.text_area(key="text").input("I absolutely love this phone, best purchase ever!").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Positive", app.success[0].value)
        self.assertEqual(len(app.get("vega_lite_chart")), 2)  # one probability chart per model

    def test_analyzer_validates_input(self):
        app = self.open("analyzer")
        app.button[0].click().run()
        self.assertIn("enter some text", app.warning[0].value)
        app.text_area(key="text").input("12345 !!!").run()
        app.button[0].click().run()
        self.assertIn("no words", app.warning[0].value)
        self.assertEqual(len(app.success), 0)

    def test_analyzer_flags_unknown_words(self):
        app = self.open("analyzer")
        app.text_area(key="text").input("qwxzv blorptang").run()
        app.button[0].click().run()
        self.assertTrue(any("vocabulary" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
