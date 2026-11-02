"""Dataset Analysis page: Phase 1 EDA results from reports/."""
import streamlit as st

from webapp import resources, ui

ui.page_header("Dataset Analysis", "What the training data looks like, from the Phase 1 exploratory analysis.")

try:
    counts = resources.class_counts()
except resources.ResourceError as err:
    st.error(str(err))
    st.stop()

DATASETS = {
    "sentiment": {
        "name": "TweetEval · sentiment",
        "url": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        "about": "English tweets from SemEval-2017 Task 4, with user names replaced by `@user`.",
        "license": "CC BY 3.0 (SemEval-2017 data); use must also follow the Twitter/X Terms of Service.",
        "findings": [
            "Classes are imbalanced (neutral 45.3%, positive 39.1%, negative 15.5% of train).",
            "The test split has a different class mix: negative tweets are 32.3% of test but 15.5% of train.",
            "About 11% of train and validation tweets contain escape sequences that never occur in test; "
            "the models decode them before vectorising.",
            "Tweets are short: median 20 words (113 characters).",
        ],
    },
    "emotion": {
        "name": "dair-ai/emotion",
        "url": "https://huggingface.co/datasets/dair-ai/emotion",
        "about": "English Twitter messages labelled with six emotions, already lowercased and stripped "
                 "of punctuation.",
        "license": "Educational and research use only (per the dataset card).",
        "findings": [
            "Strong imbalance: joy is 33.5% of train and surprise only 3.6% (9.4 : 1).",
            "Train, validation and test share the same class mix.",
            "98.7% of texts contain a form of 'feel'; class-specific emotion words are very strong cues.",
            "Median text length is 17 words, with a long tail up to 66 words.",
        ],
    },
}

st.subheader("Dataset sizes")
columns = st.columns(2)
for column, (task, info) in zip(columns, DATASETS.items()):
    totals = counts[task].sum()
    with column, st.container(border=True):
        st.markdown(f"**{info['name']}** · {len(counts[task])} classes")
        a, b, c = st.columns(3)
        a.metric("Train", f"{totals['train']:,}")
        b.metric("Validation", f"{totals['validation']:,}")
        c.metric("Test", f"{totals['test']:,}")

CHARTS = [
    ("class_distribution", "Class distribution by split"),
    ("text_length", "Text length"),
    ("length_by_class", "Words per text by class"),
    ("distinctive_words", "Most distinctive words per class"),
    ("vocabulary_coverage", "Vocabulary coverage"),
]

for tab, (task, info) in zip(st.tabs([info["name"] for info in DATASETS.values()]), DATASETS.items()):
    with tab:
        st.markdown(f"{info['about']} [Dataset card]({info['url']}).")
        table = counts[task].copy()
        shares = table / table.sum() * 100
        for split in table.columns:
            table[split] = [f"{n:,} ({s:.1f}%)" for n, s in zip(counts[task][split], shares[split])]
        table.index = [ui.pretty(label) for label in table.index]
        st.markdown("**Examples per class**")
        st.dataframe(table, width="stretch")
        st.markdown("**Key findings**")
        st.markdown("\n".join(f"- {finding}" for finding in info["findings"]))
        for name, title in CHARTS:
            st.markdown(f"**{title}**")
            try:
                st.image(str(resources.figure(f"{task}_{name}")), width="stretch")
            except resources.ResourceError as err:
                st.warning(str(err))
        st.caption(f"License: {info['license']}")

st.subheader("Sources")
st.markdown(
    "- Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L. and Neves, L. (2020). *TweetEval: Unified "
    "Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP 2020.\n"
    "- Rosenthal, S., Farra, N. and Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in "
    "Twitter.* Proceedings of SemEval-2017.\n"
    "- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J. and Chen, Y.-S. (2018). *CARER: Contextualized "
    "Affect Representations for Emotion Recognition.* Proceedings of EMNLP 2018.\n\n"
    "Full statistics: `reports/dataset_summary.md` in the repository."
)
