"""Text preprocessing shared by training and inference.

Every TF-IDF vectorizer is built by build_vectorizer() with clean_text() as its
preprocessor, so the saved artifacts apply exactly the same cleaning to new text
at prediction time. Bump PREPROCESSING_VERSION whenever clean_text() or
TOKEN_PATTERN changes; saved models record the version they were trained with.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from eda import decode_artifacts

PREPROCESSING_VERSION = 1

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")

# Negations are kept as the word "not" so they survive tokenisation:
# "don't" would otherwise become "don" + "t", and the emotion data writes "dont".
CONTRACTED_NEGATIONS = {"can't": "can not", "won't": "will not", "ain't": "is not"}
NEGATION_SUFFIX_RE = re.compile(r"n't\b")
APOSTROPHE_FREE_NEGATIONS = {
    "dont": "do not", "didnt": "did not", "doesnt": "does not", "cant": "can not",
    "wont": "will not", "isnt": "is not", "wasnt": "was not", "arent": "are not",
    "werent": "were not", "couldnt": "could not", "wouldnt": "would not",
    "shouldnt": "should not", "havent": "have not", "hasnt": "has not", "hadnt": "had not",
    "aint": "is not",
}
NEGATION_WORD_RE = re.compile(r"\b(" + "|".join(APOSTROPHE_FREE_NEGATIONS) + r")\b")

# Tokens: words of two or more characters, "!" and "?", simple emoticons such as
# :) ;-( :d (text is lowercased first), and emoji characters.
EMOJI_CLASS = "[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]"
TOKEN_PATTERN = r"(?u)\b\w\w+\b|[!?]|[:;=]-?[()dp]|" + EMOJI_CLASS


def clean_text(text: str) -> str:
    """Normalise one text for TF-IDF; used identically for training and prediction.

    Decodes escape sequences and HTML entities (Phase 1's decode_artifacts),
    lowercases, replaces URLs and @mentions with placeholder words, strips the
    "#" from hashtags and spells negations out as "not". Punctuation that the
    token pattern keeps ("!", "?", emoticons) is left in place.
    """
    text = decode_artifacts(str(text)).replace("’", "'").lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" mention ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    for contraction, expanded in CONTRACTED_NEGATIONS.items():
        text = text.replace(contraction, expanded)
    text = NEGATION_SUFFIX_RE.sub(" not", text)
    text = NEGATION_WORD_RE.sub(lambda m: APOSTROPHE_FREE_NEGATIONS[m.group(1)], text)
    return WHITESPACE_RE.sub(" ", text).strip()


def build_vectorizer(ngram_range=(1, 1), min_df=2, max_df=0.95, max_features=None) -> TfidfVectorizer:
    """TF-IDF vectorizer with the project's preprocessing and tokenisation.

    No stop-word list is used, so negations such as "not", "no" and "never" are kept.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=TOKEN_PATTERN,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
