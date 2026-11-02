"""Write reports/dataset_summary.md from the downloaded data in data/raw/.

    python src/make_dataset_summary.py

Run src/download_data.py first. The figures linked from the report are
produced by the EDA notebooks in notebooks/.
"""
from datetime import date

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import eda
from download_data import DATASETS

REPORT_PATH = eda.ROOT / "reports" / "dataset_summary.md"
NEGATIONS = ["not", "no", "nor", "never", "nothing", "none", "cannot"]

INFO = {
    "sentiment": {
        "heading": "Task A: Sentiment analysis (TweetEval)",
        "origin": "English tweets from SemEval-2017 Task 4 (subtask A), packaged by the TweetEval "
                  "benchmark. User names are replaced with `@user`. TweetEval's official metric for "
                  "this task is macro-averaged recall.",
        "license": "CC BY 3.0 (SemEval-2017 data); use must also comply with the Twitter/X Terms of Service",
        "notebook": "notebooks/sentiment_eda.ipynb",
    },
    "emotion": {
        "heading": "Task B: Emotion detection (dair-ai Emotion)",
        "origin": "English Twitter messages labelled with six basic emotions, from the CARER paper "
                  "(Saravia et al., 2018). The texts are distributed already lowercased, with "
                  "punctuation, digits, hashtags and mentions removed.",
        "license": "Educational and research use only (per the dataset card)",
        "notebook": "notebooks/emotion_eda.ipynb",
    },
}

FIGURES = [
    ("class_distribution", "Class share by split"),
    ("text_length", "Text length"),
    ("length_by_class", "Words per text by class"),
    ("vocabulary_coverage", "Vocabulary coverage"),
    ("distinctive_words", "Most distinctive words per class"),
]


# TEXT_PATTERNS names -> wording used in the observations
SURFACE_FEATURES = {
    "uppercase letter": "uppercase letters",
    "number": "numbers",
    "! or ?": "`!` or `?`",
    "apostrophe": "apostrophes",
    "hashtag": "hashtags",
    "@user mention": "`@user` mentions",
}


def join_words(items: list[str]) -> str:
    """'a', 'a or b', 'a, b or c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def table(header: list[str], rows: list[list], align: str) -> str:
    """Markdown table; align holds one 'l' or 'r' per column."""
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def md_escape(text: str) -> str:
    return text.replace("&", "&amp;")


def analyse(key: str) -> dict:
    splits = eda.load_splits(key)
    labels = eda.label_order(splits)
    train = eda.add_length_columns(splits["train"])
    counts = eda.class_counts(splits, labels)
    return {
        "labels": labels,
        "counts": counts,
        "shares": eda.class_shares(counts),
        "quality": eda.quality_checks(splits),
        "overlap": eda.split_overlap(splits),
        "multi_split_texts": eda.texts_in_several_splits(splits),
        "imbalance": eda.imbalance_ratio(counts),
        "shift_test": eda.distribution_shift(counts, "train", "test"),
        "shift_val": eda.distribution_shift(counts, "train", "validation"),
        "length": eda.length_summary(train).iloc[0],
        "length_by_class": eda.length_summary(train, labels),
        "patterns": eda.pattern_shares(train, labels),
        "escaped_pct": {s: df["text"].str.contains(eda.TEXT_PATTERNS[eda.ESCAPED]).mean() * 100
                        for s, df in splits.items()},
        "vocab": eda.vocabulary_summary(eda.word_counts(train["text"])),
        "distinctive": eda.distinctive_words(train, labels),
    }


def glance_column(key: str, r: dict) -> dict[str, str]:
    """One dataset's column of the at-a-glance table, keyed by row name."""
    spec, counts, q = DATASETS[key], r["counts"], r["quality"]
    train, p = counts["train"], r["patterns"]["all"]
    return {
        "Hugging Face dataset": f"[{spec['path']}](https://huggingface.co/datasets/{spec['path']}), "
                                f"config `{spec['config']}`",
        "Pinned revision": f"`{spec['revision'][:7]}`",
        "Classes": f"{len(r['labels'])}: " + ", ".join(r["labels"]),
        "Train / validation / test": " / ".join(f"{counts[s].sum():,}" for s in eda.SPLITS),
        "Largest : smallest class (train)": f"{r['imbalance']:.1f} : 1 ({train.idxmax()} : {train.idxmin()})",
        "Train vs test class shift (total variation)": f"{r['shift_test']:.1f} points",
        "Median words per text (train)": f"{r['length']['words_median']:.0f}",
        "Distinct words (train)": f"{r['vocab']['vocabulary']:,}",
        "Words covering 95% of tokens (train)": f"{r['vocab']['words_for_95pct']:,}",
        "Repeated texts / texts with conflicting labels (train)":
            f"{q.loc['train', 'repeated_texts']} / {q.loc['train', 'conflicting_label_texts']}",
        "Distinct texts found in more than one split": f"{r['multi_split_texts']}",
        "Texts with uppercase / `!` or `?` / `@user` / hashtag":
            f"{p['uppercase letter']:.0f}% / {p['! or ?']:.0f}% / {p['@user mention']:.0f}% / {p['hashtag']:.0f}%",
        "License": INFO[key]["license"],
    }


def at_a_glance(results: dict) -> str:
    columns = {key: glance_column(key, r) for key, r in results.items()}
    rows = [[name, columns["sentiment"][name], columns["emotion"][name]] for name in columns["sentiment"]]
    return table(["", "Task A: Sentiment", "Task B: Emotion"], rows, "lll")


def observations(r: dict) -> list[str]:
    shares, train_share = r["shares"], r["shares"]["train"]
    big, small = train_share.idxmax(), train_share.idxmin()
    notes = [
        f"**Imbalanced classes.** `{big}` is the largest class ({train_share[big]:.1f}% of train) and "
        f"`{small}` the smallest ({train_share[small]:.1f}%), a {r['imbalance']:.1f} : 1 ratio."
    ]

    if r["shift_test"] >= 5:
        diff = shares["test"] - shares["train"]
        up, down = diff.idxmax(), diff.idxmin()
        notes.append(
            f"**The test split has a different class mix.** `{up}` rises from {shares['train'][up]:.1f}% "
            f"of train to {shares['test'][up]:.1f}% of test, while `{down}` falls from "
            f"{shares['train'][down]:.1f}% to {shares['test'][down]:.1f}% (total variation distance "
            f"{r['shift_test']:.1f} points, against {r['shift_val']:.1f} between train and validation). "
            f"Validation results will not reflect the test class mix, so accuracy alone would be misleading."
        )
    else:
        notes.append(
            f"**Consistent splits.** Class shares differ by only {r['shift_test']:.1f} points between train "
            f"and test, and {r['shift_val']:.1f} between train and validation."
        )

    q = r["quality"]
    empty = int(q["empty_texts"].sum())
    shared = r["multi_split_texts"]
    notes.append(
        f"**Data quality.** {'No' if empty == 0 else empty} empty texts. Train contains "
        f"{q.loc['train', 'repeated_texts']} repeated copies of texts, and "
        f"{q.loc['train', 'conflicting_label_texts']} distinct texts appear with more than one label; "
        + (f"{shared} distinct texts occur in more than one split (possible leakage)."
           if shared else "no text occurs in more than one split.")
    )

    length, by_class = r["length"], r["length_by_class"]
    notes.append(
        f"**Short texts.** Median {length['words_median']:.0f} words ({length['chars_median']:.0f} "
        f"characters); 95% of texts have at most {length['words_p95']:.0f} words (longest "
        f"{length['words_max']:.0f}). Median length per class only ranges from "
        f"{by_class['words_median'].min():.0f} to {by_class['words_median'].max():.0f} words, so length "
        f"alone carries little class signal."
    )

    p = r["patterns"]
    if p.loc["uppercase letter", "all"] < 1:
        absent = [prose for name, prose in SURFACE_FEATURES.items() if p.loc[name, "all"] == 0]
        notes.append(
            "**Pre-cleaned text.** "
            + (f"No train text contains {join_words(absent)}. " if absent else "")
            + f"{p.loc['feel (any form)', 'all']:.1f}% of texts contain a form of 'feel', so 'feel' and "
            f"'feeling' carry no class information."
        )
    else:
        bang = p.loc["! or ?", r["labels"]]
        notes.append(
            f"**Raw tweet text.** {p.loc['uppercase letter', 'all']:.0f}% of texts contain uppercase "
            f"letters, {p.loc['@user mention', 'all']:.0f}% an `@user` mention, "
            f"{p.loc['hashtag', 'all']:.0f}% a hashtag and {p.loc['HTML entity (e.g. &amp;)', 'all']:.0f}% "
            f"an HTML entity such as `&amp;`; URLs are almost absent ({p.loc['URL or http', 'all']:.1f}%). "
            f"`!` or `?` appears in {bang.max():.1f}% of `{bang.idxmax()}` texts but only "
            f"{bang.min():.1f}% of `{bang.idxmin()}` texts."
        )
    esc = r["escaped_pct"]
    if esc["train"] >= 1:
        notes.append(
            f"**Train and test are formatted differently.** {esc['train']:.1f}% of train and "
            f"{esc['validation']:.1f}% of validation texts contain literal escape sequences such as "
            f"`\\u2019` (’) and `\\u002c` (,), against {esc['test']:.1f}% of test texts. Left as they "
            f"are, a bag-of-words model would learn junk tokens such as `u2019t` and `u002c` that never "
            f"occur at test time."
        )

    v = r["vocab"]
    notes.append(
        f"**Vocabulary.** {v['vocabulary']:,} distinct words in train, {v['seen_once'] / v['vocabulary']:.0%} "
        f"of them seen only once; the {v['words_for_95pct']:,} most frequent words cover 95% of all word "
        f"occurrences."
    )

    strongest = {lab: t["ratio"].iloc[0] for lab, t in r["distinctive"].items()}
    examples = "; ".join(
        f"`{lab}`: " + ", ".join(r["distinctive"][lab]["word"].head(3)) for lab in r["labels"]
    )
    weakest = min(strongest, key=strongest.get)
    others = [v for k, v in strongest.items() if k != weakest]
    if strongest[weakest] < 0.5 * min(others):
        notes.append(
            f"**Indicative words.** The most distinctive words per class are {examples}. `{weakest}` has "
            f"no strongly indicative words (its strongest is only {strongest[weakest]:.1f}× more frequent "
            f"than in the other classes, against {min(others):.1f}×–{max(others):.1f}× for the rest), so it "
            f"is likely to be the hardest class to separate."
        )
    else:
        notes.append(
            f"**Indicative words.** Every class has words that occur almost only in that class (strongest "
            f"ratios {min(strongest.values()):.0f}×–{max(strongest.values()):.0f}×), e.g. {examples}. A "
            f"bag-of-words model should separate the classes well, but may lean on these keywords rather "
            f"than on context."
        )
    return notes


def dataset_section(key: str, r: dict) -> str:
    info, spec = INFO[key], DATASETS[key]
    counts, shares, q = r["counts"], r["shares"], r["quality"]
    out = [f"## {info['heading']}", ""]
    out += [
        f"- **Source:** [{spec['path']}](https://huggingface.co/datasets/{spec['path']}), "
        f"config `{spec['config']}`, revision `{spec['revision'][:7]}`",
        f"- **Origin:** {info['origin']}",
        f"- **License:** {info['license']}",
        f"- **Notebook:** [`{info['notebook']}`](../{info['notebook']})",
        "",
        "### Class distribution",
        "",
    ]
    rows = [[f"{lab} ({i})"] + [f"{counts.loc[lab, s]:,} ({shares.loc[lab, s]:.1f}%)" for s in eda.SPLITS]
            for i, lab in enumerate(r["labels"])]
    rows.append(["**Total**"] + [f"**{counts[s].sum():,}**" for s in eda.SPLITS])
    out += [table(["Class (label id)", "Train", "Validation", "Test"], rows, "lrrr"), ""]

    out += ["### Data quality", ""]
    rows = [[s, f"{q.loc[s, 'rows']:,}", q.loc[s, "empty_texts"], q.loc[s, "repeated_texts"],
             q.loc[s, "conflicting_label_texts"]] for s in eda.SPLITS]
    out += [table(["Split", "Rows", "Empty texts", "Repeated texts", "Texts with conflicting labels"],
                  rows, "lrrrr"), ""]
    overlap = ", ".join(f"{pair}: {n}" for pair, n in r["overlap"]["shared_texts"].items())
    out += [f"Texts shared between splits: {overlap}.", ""]

    length = r["length"]
    out += ["### Text length (train)", ""]
    rows = [[unit] + [f"{length[f'{prefix}_{stat}']:,.1f}" if stat == "mean" else f"{length[f'{prefix}_{stat}']:,.0f}"
                      for stat in ("mean", "median", "p95", "max")]
            for unit, prefix in (("Words", "words"), ("Characters", "chars"))]
    out += [table(["", "Mean", "Median", "95th percentile", "Max"], rows, "lrrrr"), ""]
    medians = ", ".join(f"{lab} {m:.0f}" for lab, m in r["length_by_class"]["words_median"].items())
    out += [f"Median words per class: {medians}.", ""]

    out += ["### Text features (% of train texts)", ""]
    p = r["patterns"]
    rows = [[md_escape(feature)] + [f"{p.loc[feature, c]:.1f}" for c in p.columns] for feature in p.index]
    out += [table(["Feature"] + list(p.columns), rows, "l" + "r" * len(p.columns)), ""]

    v = r["vocab"]
    out += [
        "### Vocabulary (train)",
        "",
        f"- Word tokens: {v['tokens']:,}",
        f"- Distinct words: {v['vocabulary']:,} ({v['seen_once']:,} seen only once)",
        f"- Words needed to cover 90% / 95% of tokens: {v['words_for_90pct']:,} / {v['words_for_95pct']:,}",
        "",
        "### Most distinctive words per class (train)",
        "",
        "Ratio = how many times more frequent the word is in the class than in the other classes "
        "(stop words removed; words used at least 25 times).",
        "",
    ]
    rows = [[lab, ", ".join(f"{w} ({x:.0f}×)" for w, x in zip(t["word"].head(6), t["ratio"].head(6)))]
            for lab, t in r["distinctive"].items()]
    out += [table(["Class", "Top words (ratio)"], rows, "ll"), ""]

    out += ["### Observations", ""] + [f"- {note}" for note in observations(r)] + [""]
    out += ["### Figures", ""]
    for name, title in FIGURES:
        out += [f"![{title}](figures/{key}_{name}.png)", ""]
    return "\n".join(out)


def implications(results: dict) -> str:
    s, e = results["sentiment"], results["emotion"]
    negations = [w for w in NEGATIONS if w in ENGLISH_STOP_WORDS]
    conflicts = [r["quality"].loc["train", "conflicting_label_texts"] for r in (s, e)]
    bullets = [
        "**Splits.** Keep the official train / validation / test splits: tune on validation and "
        "evaluate on test once, so results are comparable with published numbers.",
        "**Metrics.** Report macro-averaged F1 with per-class precision, recall and a confusion matrix "
        "(plus macro-averaged recall for TweetEval). Accuracy alone is misleading with imbalanced classes "
        f"and, for sentiment, a test split whose class mix differs from train ({s['shift_test']:.1f} points).",
        "**Class imbalance.** Compare `LogisticRegression(class_weight=None)` with "
        f"`class_weight=\"balanced\"`, watching the smallest classes (`{s['counts']['train'].idxmin()}` "
        f"for sentiment, `{e['counts']['train'].idxmin()}` for emotion).",
        f"**Cleaning.** Sentiment text is raw: decode the literal `\\uXXXX` escape sequences "
        f"({s['escaped_pct']['train']:.0f}% of train texts, {s['escaped_pct']['test']:.1f}% of test) and HTML "
        "entities such as `&amp;`, lowercase it, and treat `@user` as a single token or drop it. "
        "`src/eda.py`'s `decode_artifacts()` already does the decoding. The emotion text is already clean.",
        "**Stop words.** Do not use scikit-learn's built-in English stop-word list for sentiment: it "
        f"removes negations ({', '.join(negations)}). TF-IDF's IDF term already down-weights words "
        "that occur in every class (such as 'feel' in the emotion data).",
        f"**Duplicates.** Drop texts with conflicting labels from the training split ({conflicts[0]} for "
        f"sentiment, {conflicts[1]} for emotion); leave validation and test unchanged.",
        "**TF-IDF starting point.** Word unigrams and bigrams, `min_df=2` (about half of all distinct "
        "words occur only once), `sublinear_tf=True`, and `max_features` near the 95%-coverage points "
        f"({s['vocab']['words_for_95pct']:,} for sentiment, {e['vocab']['words_for_95pct']:,} for "
        "emotion), then tune on validation.",
    ]
    return "\n".join(["## Implications for Phase 2 (modelling)", ""] + [f"- {b}" for b in bullets] + [""])


def build_report() -> str:
    results = {key: analyse(key) for key in INFO}
    parts = [
        "# Dataset Summary",
        "",
        f"_Phase 1 of **Sentiment and Emotion Analysis Using Machine Learning**. Generated by "
        f"`src/make_dataset_summary.py` on {date.today().isoformat()} from the files in `data/raw/`; "
        f"the figures come from the EDA notebooks._",
        "",
        "## At a glance",
        "",
        at_a_glance(results),
        "",
    ]
    parts += [dataset_section(key, results[key]) for key in INFO]
    parts.append(implications(results))
    parts += [
        "## Reproducing this report",
        "",
        "```powershell",
        "python src/download_data.py",
        "jupyter nbconvert --to notebook --execute --inplace notebooks/sentiment_eda.ipynb notebooks/emotion_eda.ipynb",
        "python src/make_dataset_summary.py",
        "```",
        "",
        "## Sources",
        "",
        "- Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L. and Neves, L. (2020). *TweetEval: Unified "
        "Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP 2020.",
        "- Rosenthal, S., Farra, N. and Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in "
        "Twitter.* Proceedings of SemEval-2017, pp. 502-518.",
        "- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J. and Chen, Y.-S. (2018). *CARER: Contextualized "
        "Affect Representations for Emotion Recognition.* Proceedings of EMNLP 2018, pp. 3687-3697.",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(eda.ROOT)}")


if __name__ == "__main__":
    main()
