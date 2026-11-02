"""Shared helpers for the Phase 1 exploratory data analysis (EDA).

The sentiment and emotion notebooks and src/make_dataset_summary.py all use
these functions, so both datasets are checked, measured and plotted the same way.
"""
import html
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures"
SPLITS = ["train", "validation", "test"]

# Chart colours: a light surface, text inks, hairline chrome, and the first three
# slots of a colour-blind-checked categorical palette - one per split, always in
# this order. Single-series charts use the train colour (they plot train data).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}
ACCENT = SPLIT_COLORS["train"]
BAR_INCHES = 0.16  # bar thickness cap (~24 px in the 150 dpi saved figures)

# Share of texts matching each pattern is reported per class.
ESCAPED = "escaped character (e.g. \\u2019)"
TEXT_PATTERNS = {
    "@user mention": r"@user",
    "hashtag": r"#\w",
    "URL or http": r"http",
    "HTML entity (e.g. &amp;)": r"&\w+;",
    ESCAPED: r"\\u[0-9a-fA-F]{4}",  # a literal backslash, "u" and four hex digits
    "number": r"\b\d",
    "uppercase letter": r"[A-Z]",
    "! or ?": r"[!?]",
    "apostrophe": r"'",
    "feel (any form)": r"\bfeel",
}

ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
TOKEN_RE = re.compile(r"[a-z]+")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | {
    # dataset artefacts: the @user placeholder, HTML entities, URL fragments
    "user", "amp", "quot", "http", "https", "com",
    # contractions once apostrophes are removed
    "im", "ive", "dont", "didnt", "doesnt", "cant", "wont", "isnt", "wasnt",
    "arent", "thats", "youre", "theyre", "lets",
}


# --- Loading -----------------------------------------------------------------

def load_splits(dataset: str) -> dict[str, pd.DataFrame]:
    """Load data/raw/<dataset>/{train,validation,test}.csv written by download_data.py."""
    folder = RAW_DIR / dataset
    missing = [s for s in SPLITS if not (folder / f"{s}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {folder}. Run: python src/download_data.py --dataset {dataset}"
        )
    # keep_default_na=False stops texts such as "NA" or "null" being read as missing.
    return {s: pd.read_csv(folder / f"{s}.csv", keep_default_na=False) for s in SPLITS}


def label_order(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Class names ordered by their integer label id."""
    pairs = splits["train"][["label", "label_name"]].drop_duplicates().sort_values("label")
    return pairs["label_name"].tolist()


# --- Data quality --------------------------------------------------------------

def quality_checks(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Empty texts, repeated texts and texts that carry more than one label, per split."""
    rows = {}
    for split, df in splits.items():
        text = df["text"]
        rows[split] = {
            "rows": len(df),
            "empty_texts": int((text.str.strip() == "").sum()),
            "repeated_texts": int(text.duplicated().sum()),
            "conflicting_label_texts": int(df.groupby("text")["label"].nunique().gt(1).sum()),
        }
    return pd.DataFrame(rows).T.rename_axis("split")


def conflicting_examples(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Rows whose exact text also appears with a different label."""
    conflicted = df.groupby("text")["label"].transform("nunique") > 1
    return df.loc[conflicted, ["text", "label_name"]].sort_values("text").head(n)


def split_overlap(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Number of distinct texts shared by each pair of splits (possible leakage)."""
    texts = {s: set(df["text"]) for s, df in splits.items()}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    return pd.DataFrame(
        {"shared_texts": [len(texts[a] & texts[b]) for a, b in pairs]},
        index=pd.Index([f"{a} & {b}" for a, b in pairs], name="splits"),
    )


def texts_in_several_splits(splits: dict[str, pd.DataFrame]) -> int:
    """Distinct texts that occur in two or more splits."""
    seen = Counter(t for df in splits.values() for t in set(df["text"]))
    return sum(1 for n in seen.values() if n > 1)


# --- Class distribution --------------------------------------------------------

def class_counts(splits: dict[str, pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    """Examples per class (rows, in label order) and split (columns)."""
    counts = pd.DataFrame({s: df["label_name"].value_counts() for s, df in splits.items()})
    return counts.reindex(labels).fillna(0).astype(int).rename_axis("class")


def class_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Percentage of each split that belongs to each class."""
    return (counts / counts.sum() * 100).round(1)


def imbalance_ratio(counts: pd.DataFrame, split: str = "train") -> float:
    """Largest class size divided by smallest class size."""
    return float(counts[split].max() / counts[split].min())


def distribution_shift(counts: pd.DataFrame, a: str = "train", b: str = "test") -> float:
    """Total variation distance between two splits' class distributions, in percentage points."""
    p = counts[a] / counts[a].sum()
    q = counts[b] / counts[b].sum()
    return float((p - q).abs().sum() / 2 * 100)


# --- Text length and surface features ------------------------------------------

def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with n_chars and n_words (whitespace-separated) columns."""
    out = df.copy()
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].str.split().str.len()
    return out


def _length_stats(frame: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "words_mean": frame["n_words"].mean(),
        "words_median": frame["n_words"].median(),
        "words_p95": frame["n_words"].quantile(0.95),
        "words_max": frame["n_words"].max(),
        "chars_mean": frame["n_chars"].mean(),
        "chars_median": frame["n_chars"].median(),
        "chars_p95": frame["n_chars"].quantile(0.95),
        "chars_max": frame["n_chars"].max(),
    })


def length_summary(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Length statistics for all texts, or per class when labels are given."""
    if labels is None:
        return _length_stats(df).to_frame("all").T.round(1)
    rows = {lab: _length_stats(df[df["label_name"] == lab]) for lab in labels}
    return pd.DataFrame(rows).T.rename_axis("class").round(1)


def pattern_shares(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Percentage of texts in each class that match each TEXT_PATTERNS entry."""
    flags = pd.DataFrame({
        name: df["text"].str.contains(pattern, regex=True) for name, pattern in TEXT_PATTERNS.items()
    })
    table = flags.groupby(df["label_name"]).mean().T.reindex(columns=labels) * 100
    table["all"] = flags.mean() * 100
    return table.rename_axis("feature").round(1)


# --- Vocabulary ------------------------------------------------------------------

def decode_artifacts(text: str) -> str:
    """Decode literal \\uXXXX escapes (e.g. \\u002c -> ",") and HTML entities (&amp; -> &)."""
    return html.unescape(ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text))


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens; apostrophes are dropped so "don't" becomes "dont"."""
    text = decode_artifacts(text).lower()
    return TOKEN_RE.findall(text.replace("'", "").replace("’", ""))


def content_words(text: str) -> list[str]:
    """Tokens of three or more letters that are not stop words."""
    return [t for t in tokenize(text) if len(t) > 2 and t not in STOP_WORDS]


def word_counts(texts) -> Counter:
    """Frequency of every token across texts."""
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts


def vocabulary_summary(counts: Counter) -> dict[str, int]:
    """Token and vocabulary sizes, words seen once, and words needed to cover 90%/95% of tokens."""
    freq = np.sort(np.fromiter(counts.values(), dtype=np.int64))[::-1]
    covered = np.cumsum(freq) / freq.sum()
    return {
        "tokens": int(freq.sum()),
        "vocabulary": int(freq.size),
        "seen_once": int((freq == 1).sum()),
        "words_for_90pct": int(np.searchsorted(covered, 0.90) + 1),
        "words_for_95pct": int(np.searchsorted(covered, 0.95) + 1),
    }


def top_words(df: pd.DataFrame, labels: list[str], n: int = 10) -> pd.DataFrame:
    """Most frequent content words per class (stop words removed)."""
    table = {}
    for lab in labels:
        counts = Counter(w for text in df.loc[df["label_name"] == lab, "text"] for w in content_words(text))
        table[lab] = [w for w, _ in counts.most_common(n)]
    return pd.DataFrame(table, index=pd.RangeIndex(1, n + 1, name="rank"))


def distinctive_words(df: pd.DataFrame, labels: list[str], n: int = 10,
                      min_count: int = 25) -> dict[str, pd.DataFrame]:
    """Words most over-represented in each class compared with all other classes.

    ratio = (share of this class's words that are w) / (share of the other
    classes' words that are w), with add-one smoothing. Only words used at least
    min_count times in total are ranked, so rare words cannot dominate.
    """
    per_class = {lab: Counter() for lab in labels}
    for text, lab in zip(df["text"], df["label_name"]):
        per_class[lab].update(content_words(text))
    total = Counter()
    for counts in per_class.values():
        total.update(counts)
    vocab = [w for w, c in total.items() if c >= min_count]
    all_words = sum(total.values())

    tables = {}
    for lab in labels:
        in_class = per_class[lab]
        n_in = sum(in_class.values())
        n_out = all_words - n_in
        rows = [
            (w, in_class[w],
             ((in_class[w] + 1) / (n_in + len(vocab))) / ((total[w] - in_class[w] + 1) / (n_out + len(vocab))))
            for w in vocab
        ]
        table = pd.DataFrame(rows, columns=["word", "count_in_class", "ratio"])
        tables[lab] = table.nlargest(n, "ratio").reset_index(drop=True).round({"ratio": 1})
    return tables


# --- Charts ------------------------------------------------------------------------

def set_plot_style() -> None:
    """Apply the shared low-ink chart style to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.titlepad": 8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 9.5,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": BASELINE,
        "ytick.color": BASELINE,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
    })


def save_figure(fig, name: str) -> Path:
    """Save fig as reports/figures/<name>.png and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def _titles(fig, ax, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="semibold", color=INK)
    if subtitle and ax is not None:
        ax.set_title(subtitle, fontsize=9, fontweight="normal", color=INK_SECONDARY)


def _thousands(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def plot_class_distribution(counts: pd.DataFrame, title: str):
    """Grouped bars: each class's share of the train, validation and test splits."""
    shares = counts / counts.sum() * 100
    n = len(shares)
    fig, ax = plt.subplots(figsize=(2.2 + 1.1 * n, 3.6), layout="constrained")
    # Keep bars thin whatever the class count: convert the thickness cap to data units.
    width = min(0.25, BAR_INCHES * n / (fig.get_figwidth() * 0.8))
    x = np.arange(n)
    for i, split in enumerate(shares.columns):
        ax.bar(x + (i - 1) * width, shares[split], width, color=SPLIT_COLORS[split],
               edgecolor=SURFACE, linewidth=1, label=f"{split} (n = {counts[split].sum():,})")
    ax.set_xticks(x, shares.index)
    ax.tick_params(axis="x", length=0, labelcolor=INK, labelsize=9.5)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylabel("Share of the split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, borderaxespad=0.2,
              handlelength=1, handleheight=1)
    _titles(fig, None, title)
    return fig


def plot_length_distribution(df: pd.DataFrame, title: str):
    """Histograms of words and characters per text, with the median marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    panels = [("n_words", "Words per text", 1), ("n_chars", "Characters per text", 5)]
    for ax, (column, label, step) in zip(axes, panels):
        values = df[column]
        bins = np.arange(values.min() - step / 2, values.max() + step, step)
        ax.hist(values, bins=bins, color=ACCENT, edgecolor=SURFACE, linewidth=0.4)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # headroom so the median label clears the bars
        median = values.median()
        ax.axvline(median, color=INK, linewidth=1)
        ax.annotate(f"median {median:,.0f}", xy=(median, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -2), textcoords="offset points", ha="left", va="top",
                    color=INK_SECONDARY, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("Texts")
        ax.grid(axis="x", visible=False)
        _thousands(ax.yaxis)
    _titles(fig, None, title)
    return fig


def plot_length_by_class(df: pd.DataFrame, labels: list[str], title: str):
    """Horizontal box plots of words per text for each class."""
    data = [df.loc[df["label_name"] == lab, "n_words"] for lab in labels]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.42 * len(labels)), layout="constrained")
    ax.boxplot(data, orientation="horizontal", tick_labels=labels, widths=0.45, patch_artist=True,
               showfliers=False,
               boxprops={"facecolor": to_rgba(ACCENT, 0.12), "edgecolor": ACCENT, "linewidth": 1},
               medianprops={"color": ACCENT, "linewidth": 2},
               whiskerprops={"color": INK_MUTED, "linewidth": 1},
               capprops={"color": INK_MUTED, "linewidth": 1})
    ax.invert_yaxis()
    ax.set_xlabel("Words per text")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9.5)
    _titles(fig, ax, title, "Box = middle 50% of texts, line = median, whiskers = 1.5 × IQR "
                            "(outliers not drawn)")
    return fig


def plot_distinctive_words(tables: dict[str, pd.DataFrame], labels: list[str], title: str,
                           ncols: int = 3):
    """Small multiples: the most over-represented words of each class."""
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows),
                             layout="constrained", squeeze=False)
    for ax, lab in zip(axes.flat, labels):
        table = tables[lab]
        y = np.arange(len(table))
        ax.barh(y, table["ratio"], height=0.6, color=ACCENT)
        ax.set_yticks(y, table["word"])
        ax.invert_yaxis()
        ax.set_title(lab)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelcolor=INK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    fig.supxlabel("Times more frequent in this class than in the other classes",
                  fontsize=9.5, color=INK_SECONDARY)
    _titles(fig, None, title)
    return fig


def plot_vocabulary_coverage(counts: Counter, title: str):
    """Share of all tokens covered when keeping only the k most frequent words."""
    freq = np.sort(np.fromiter(counts.values(), dtype=np.int64))[::-1]
    covered = np.cumsum(freq) / freq.sum() * 100
    ranks = np.arange(1, freq.size + 1)
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(ranks, covered, color=ACCENT, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    for k in (100, 1_000, 10_000):
        if k < freq.size:
            ax.plot(k, covered[k - 1], "o", ms=7, color=ACCENT, mec=SURFACE, mew=1, zorder=3)
            ax.annotate(f"{k:,} words → {covered[k - 1]:.0f}%", (k, covered[k - 1]),
                        xytext=(8, -16), textcoords="offset points", fontsize=9, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(1, freq.size * 1.05)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    _thousands(ax.xaxis)
    ax.set_xlabel("Most frequent distinct words kept (log scale)")
    ax.set_ylabel("Word occurrences covered")
    ax.grid(axis="x", visible=False)
    seen_once = (freq == 1).sum()
    _titles(fig, ax, title, f"{freq.size:,} distinct words in the train split; "
                            f"{seen_once / freq.size:.0%} of them occur only once")
    return fig
