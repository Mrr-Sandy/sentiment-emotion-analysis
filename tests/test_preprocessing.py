import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import preprocessing  # noqa: E402

BACKSLASH = "\\"


class CleanTextTest(unittest.TestCase):
    def test_decodes_escape_sequences_and_html_entities(self):
        raw = f"I can{BACKSLASH}u2019t stop{BACKSLASH}u002c fish &amp; chips"
        self.assertEqual(preprocessing.clean_text(raw), "i can not stop, fish & chips")

    def test_keeps_negations(self):
        for raw in ["I don't like it", "i dont like it", "I didn't", "won't", "Not good", "no way, never"]:
            cleaned = preprocessing.clean_text(raw)
            self.assertTrue({"not", "no", "never"} & set(cleaned.split()), msg=f"{raw!r} -> {cleaned!r}")

    def test_normalises_mentions_urls_and_hashtags(self):
        cleaned = preprocessing.clean_text("@user loving #Summer https://t.co/x1 www.example.com")
        self.assertEqual(cleaned, "mention loving summer url url")

    def test_is_idempotent(self):
        once = preprocessing.clean_text("Hey @user, I don't LOVE this &amp; that! :)")
        self.assertEqual(preprocessing.clean_text(once), once)


class TokenisationTest(unittest.TestCase):
    def setUp(self):
        self.analyze = preprocessing.build_vectorizer(min_df=1, max_df=1.0).build_analyzer()

    def test_keeps_negation_punctuation_and_emoticons(self):
        tokens = self.analyze("This is NOT good!! Why? :( but :-) fine")
        for expected in ["not", "!", "?", ":(", ":-)"]:
            self.assertIn(expected, tokens)

    def test_keeps_emoji(self):
        emoji = chr(0x1F600)
        self.assertIn(emoji, self.analyze(f"great day {emoji}"))

    def test_uses_no_stop_word_list(self):
        self.assertIsNone(preprocessing.build_vectorizer().stop_words)


if __name__ == "__main__":
    unittest.main()
