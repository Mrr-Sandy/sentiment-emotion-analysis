"""Evaluate the saved models on the untouched test splits.

    python src/evaluate.py

Run the training scripts first. Writes, for each task:
  reports/metrics/<task>_test_metrics.json         all metrics, per-class scores, confusion matrix
  reports/metrics/<task>_classification_report.txt scikit-learn's classification report
  reports/figures/<task>_confusion_matrix.png
  reports/figures/<task>_per_class_metrics.png
plus reports/figures/model_comparison.png, reports/metrics/model_comparison.csv
and reports/model_evaluation.md.
"""
import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import eda
import modeling
from predict import load_classifier

TASKS = {"sentiment": "Sentiment (TweetEval)", "emotion": "Emotion (dair-ai)"}
REPORT_PATH = eda.ROOT / "reports" / "model_evaluation.md"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the Phase 1 palette
CONFUSION_CMAP = LinearSegmentedColormap.from_list("share", [eda.SURFACE, "#86b6ef", "#104281"])

log = logging.getLogger("evaluate")


# --- Metrics ----------------------------------------------------------------------

def evaluate_task(task: str) -> dict:
    """Predict the test split once and compute every metric from the predictions."""
    classifier = load_classifier(task)
    test = modeling.load_processed(task, "test")
    raw_test_rows = len(eda.load_splits(task)["test"])
    if len(test) != raw_test_rows:
        raise RuntimeError(f"{task}: processed test has {len(test)} rows, raw test has {raw_test_rows}")

    labels = classifier.labels
    y_true = test["label_name"].tolist()
    y_pred = classifier.predict(test["text"])
    unexpected = sorted(set(y_pred) - set(labels))
    if unexpected:
        raise RuntimeError(f"{task}: model predicted unknown classes {unexpected}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted",
                                               zero_division=0)
    return {
        "task": task,
        "test_rows": len(test),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2],
        "weighted_precision": weighted[0], "weighted_recall": weighted[1], "weighted_f1": weighted[2],
        "per_class": {lab: {"precision": p, "recall": r, "f1": f, "support": int(s)}
                      for lab, p, r, f, s in zip(labels, precision, recall, f1, support)},
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
        "classification_report": classification_report(y_true, y_pred, labels=labels, digits=4,
                                                       zero_division=0),
        "model": classifier.metadata,
    }


def save_metrics(result: dict) -> None:
    task = result["task"]
    modeling.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in result.items() if k != "classification_report"}
    (modeling.METRICS_DIR / f"{task}_test_metrics.json").write_text(json.dumps(data, indent=2),
                                                                     encoding="utf-8")
    (modeling.METRICS_DIR / f"{task}_classification_report.txt").write_text(
        result["classification_report"], encoding="utf-8")


# --- Charts --------------------------------------------------------------------------

def plot_confusion_matrix(result: dict, title: str):
    """Rows = true class, columns = predicted class; colour = share of the true class."""
    labels = result["confusion_matrix"]["labels"]
    counts = np.array(result["confusion_matrix"]["matrix"])
    shares = counts / counts.sum(axis=1, keepdims=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(2.4 + 0.95 * n, 1.6 + 0.8 * n), layout="constrained")
    image = ax.imshow(shares, cmap=CONFUSION_CMAP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ink = "#ffffff" if shares[i, j] > 0.55 else eda.INK
            ax.text(j, i, f"{shares[i, j]:.0%}\n{counts[i, j]:,}", ha="center", va="center",
                    fontsize=9, color=ink)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.tick_params(length=0, labelcolor=eda.INK)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.8, format=PercentFormatter(xmax=1, decimals=0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelcolor=eda.INK_SECONDARY)
    bar.set_label("Share of the true class", color=eda.INK_SECONDARY)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="semibold", color=eda.INK)
    ax.set_title("Each cell: share of the row's true class and number of test texts",
                 fontsize=9, fontweight="normal", color=eda.INK_SECONDARY)
    return fig


def plot_grouped_bars(table: pd.DataFrame, title: str, subtitle: str):
    """Grouped bars: one group per row of table, one series per column (values 0-1)."""
    n, k = table.shape
    fig, ax = plt.subplots(figsize=(2.6 + 1.25 * n, 3.8), layout="constrained")
    width = min(0.8 / k, eda.BAR_INCHES * n / (fig.get_figwidth() * 0.8))
    x = np.arange(n)
    for i, column in enumerate(table.columns):
        ax.bar(x + (i - (k - 1) / 2) * width, table[column], width, color=SERIES_COLORS[i],
               edgecolor=eda.SURFACE, linewidth=1, label=column)
    ax.set_xticks(x, table.index)
    ax.tick_params(axis="x", length=0, labelcolor=eda.INK, labelsize=9.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=k, borderaxespad=0.2,
              handlelength=1, handleheight=1)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="semibold", color=eda.INK)
    fig.supxlabel(subtitle, fontsize=9, color=eda.INK_SECONDARY)
    return fig


def per_class_table(result: dict) -> pd.DataFrame:
    rows = {lab: [m["precision"], m["recall"], m["f1"]] for lab, m in result["per_class"].items()}
    return pd.DataFrame(rows, index=["Precision", "Recall", "F1-score"]).T


def comparison_table(results: dict) -> pd.DataFrame:
    metrics = {"Accuracy": "accuracy", "Macro precision": "macro_precision",
               "Macro recall": "macro_recall", "Macro F1": "macro_f1", "Weighted F1": "weighted_f1"}
    return pd.DataFrame({TASKS[t]: [r[key] for key in metrics.values()] for t, r in results.items()},
                        index=list(metrics))


def save_figures(results: dict) -> None:
    eda.set_plot_style()
    for task, result in results.items():
        name = TASKS[task]
        fig = plot_confusion_matrix(result, f"{name}: confusion matrix (test split)")
        eda.save_figure(fig, f"{task}_confusion_matrix")
        fig = plot_grouped_bars(per_class_table(result), f"{name}: per-class scores (test split)",
                                "Precision, recall and F1-score of each class on the test split")
        eda.save_figure(fig, f"{task}_per_class_metrics")
        plt.close("all")
    fig = plot_grouped_bars(comparison_table(results), "Model performance on the test splits",
                            "TF-IDF + Logistic Regression; macro scores weight every class equally")
    eda.save_figure(fig, "model_comparison")
    plt.close("all")


# --- Report --------------------------------------------------------------------------

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def md_table(header: list[str], rows: list[list], align: str) -> str:
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def top_confusions(result: dict, n: int = 3) -> list[tuple[str, str, float, int]]:
    labels = result["confusion_matrix"]["labels"]
    counts = np.array(result["confusion_matrix"]["matrix"])
    shares = counts / counts.sum(axis=1, keepdims=True)
    pairs = [(labels[i], labels[j], shares[i, j], counts[i, j])
             for i in range(len(labels)) for j in range(len(labels)) if i != j]
    return sorted(pairs, key=lambda p: p[2], reverse=True)[:n]


def observations(task: str, result: dict) -> list[str]:
    model, per_class = result["model"], result["per_class"]
    val_f1, test_f1 = model["validation"]["macro_f1"], result["macro_f1"]
    by_weight = model["validation"]["best_macro_f1_by_class_weight"]
    worst = min(per_class, key=lambda c: per_class[c]["f1"])
    best = max(per_class, key=lambda c: per_class[c]["f1"])
    confusions = "; ".join(f"`{a}` predicted as `{b}` {s:.0%} of the time ({c:,} texts)"
                           for a, b, s, c in top_confusions(result))
    notes = [
        f"Test macro F1 is {pct(test_f1)} against {pct(val_f1)} on validation "
        f"({(test_f1 - val_f1) * 100:+.1f} points).",
        f"Balanced class weights won model selection: best validation macro F1 "
        f"{pct(by_weight.get('balanced', 0))} balanced vs {pct(by_weight.get('none', 0))} unweighted.",
        f"Best class: `{best}` (F1 {pct(per_class[best]['f1'])}); weakest: `{worst}` "
        f"(F1 {pct(per_class[worst]['f1'])}, recall {pct(per_class[worst]['recall'])}).",
        f"Most frequent confusions: {confusions}.",
    ]
    if task == "sentiment":
        notes.append(
            "Phase 1 found that the test split has twice the share of negative tweets as train (32.3% vs "
            "15.5%) and that `neutral` has no strong cue words; the confusion matrix shows where this "
            "costs accuracy. Macro recall is TweetEval's official metric for this task."
        )
    else:
        notes.append(
            "Phase 1 found that each emotion has strongly indicative keywords and that the conflicting "
            "duplicates were mostly joy/love and fear/surprise, the same pairs that remain hardest here."
        )
    return notes


def task_section(task: str, result: dict) -> str:
    model = result["model"]
    tfidf, lr = model["tfidf"], model["logistic_regression"]
    out = [f"## {TASKS[task]}", ""]
    out += [
        f"**Selected model:** TF-IDF `{tfidf['name']}` (n-grams {tuple(tfidf['ngram_range'])}, "
        f"min_df={tfidf['min_df']}, max_df={tfidf['max_df']}, {tfidf['n_features']:,} features, "
        f"sublinear TF) + Logistic Regression (C={lr['C']}, class_weight={lr['class_weight']}, "
        f"solver={lr['solver']}). Trained on {model['data']['train_rows']:,} texts; evaluated on "
        f"{result['test_rows']:,} test texts.",
        "",
        md_table(["Metric", "Validation", "Test"], [
            ["Accuracy", pct(model["validation"]["accuracy"]), pct(result["accuracy"])],
            ["Macro precision", "", pct(result["macro_precision"])],
            ["Macro recall", "", pct(result["macro_recall"])],
            ["**Macro F1**", f"**{pct(model['validation']['macro_f1'])}**", f"**{pct(result['macro_f1'])}**"],
            ["Weighted F1", pct(model["validation"]["weighted_f1"]), pct(result["weighted_f1"])],
        ], "lrr"),
        "",
        "### Per-class results (test)",
        "",
        md_table(["Class", "Precision", "Recall", "F1-score", "Support"], [
            [c, pct(m["precision"]), pct(m["recall"]), pct(m["f1"]), f"{m['support']:,}"]
            for c, m in result["per_class"].items()
        ], "lrrrr"),
        "",
        "### Confusion matrix (test; rows = true class, columns = predicted class)",
        "",
    ]
    labels = result["confusion_matrix"]["labels"]
    out += [md_table(["True \\ Predicted"] + labels,
                     [[lab] + [f"{v:,}" for v in row] for lab, row in zip(labels, result["confusion_matrix"]["matrix"])],
                     "l" + "r" * len(labels)), ""]
    out += [f"![{TASKS[task]} confusion matrix](figures/{task}_confusion_matrix.png)", "",
            f"![{TASKS[task]} per-class scores](figures/{task}_per_class_metrics.png)", ""]
    out += ["### Observations", ""] + [f"- {note}" for note in observations(task, result)] + [""]
    return "\n".join(out)


def validation_section(task: str) -> str:
    grid = pd.read_csv(modeling.METRICS_DIR / f"{task}_validation_results.csv")
    best = (grid.sort_values("val_macro_f1", ascending=False)
            .groupby(["tfidf", "class_weight"], sort=False).head(1)
            .sort_values("val_macro_f1", ascending=False))
    rows = [[r.tfidf, f"{r.n_features:,}", r.class_weight, r.C, pct(r.val_macro_f1), pct(r.val_accuracy)]
            for r in best.itertuples()]
    return "\n".join([
        f"### {TASKS[task]}: best validation result per configuration",
        "",
        md_table(["TF-IDF", "Features", "class_weight", "Best C", "Macro F1", "Accuracy"], rows, "lrlrrr"),
        "",
        f"All {len(grid)} candidates: `reports/metrics/{task}_validation_results.csv`.",
        "",
    ])


def write_report(results: dict) -> None:
    comparison = comparison_table(results)
    parts = [
        "# Model Evaluation",
        "",
        "_Phase 2 of **Sentiment and Emotion Analysis Using Machine Learning**. Generated by "
        "`src/evaluate.py` from the saved models; every number below comes from running them on the "
        "data._",
        "",
        "## Method",
        "",
        "- **Data:** the Phase 1 datasets at their pinned revisions and official splits. "
        "`src/prepare_data.py` removes training texts with conflicting labels and keeps one copy of "
        "exact duplicates; validation and test are used unchanged.",
        "- **Preprocessing** (`src/preprocessing.py`, built into each saved vectorizer so training and "
        "prediction match): decode escape sequences and HTML entities (Phase 1's `decode_artifacts`), "
        "lowercase, replace URLs and @mentions with placeholder words, strip `#` from hashtags, spell "
        "negations out as `not`. No stop-word list, so *not*, *no* and *never* are kept; `!`, `?` and "
        "emoticons are kept as tokens.",
        "- **Features:** TF-IDF with sublinear term frequency, fitted on the training split only. "
        f"Configurations tried: {', '.join(f'`{name}`' for name in modeling.TFIDF_CONFIGS)} "
        "(all with min_df=2, max_df=0.95).",
        f"- **Classifier:** Logistic Regression (lbfgs, max_iter={modeling.MAX_ITER}, "
        f"random_state={modeling.SEED}) with C in {modeling.C_VALUES} and class_weight None or "
        "\"balanced\".",
        "- **Selection:** the candidate with the highest **validation macro F1** was refitted on the "
        "training split. The test split was used once, for the results below.",
        "",
        "## Results at a glance (test splits)",
        "",
        md_table(["Metric"] + list(comparison.columns),
                 [[metric] + [pct(v) for v in row] for metric, row in comparison.iterrows()], "lrr"),
        "",
        "![Model comparison](figures/model_comparison.png)",
        "",
    ]
    parts += [task_section(task, result) for task, result in results.items()]
    parts += ["## Model selection on validation", ""] + [validation_section(task) for task in results]
    parts += [
        "## Limitations",
        "",
        "- Bag-of-words features ignore word order beyond bigrams, so sarcasm and long-range negation "
        "are hard to capture.",
        "- The models only know words seen in training; emoji support exists in the tokenizer, but "
        "neither dataset contains emoji, so they carry no learned weight.",
        "- The emotion data was labelled around emotion keywords, so its high scores may not transfer "
        "to text that expresses emotion without them.",
    ]
    for task in results:
        grid = pd.read_csv(modeling.METRICS_DIR / f"{task}_validation_results.csv")
        unweighted = grid[grid["class_weight"] == "none"].iloc[0]
        if unweighted["C"] == max(modeling.C_VALUES):
            parts.append(
                f"- {TASKS[task]}: the best unweighted model sits at the largest C tried "
                f"({unweighted['C']}, validation macro F1 {pct(unweighted['val_macro_f1'])}), so a wider grid "
                f"could narrow its gap to the selected balanced model."
            )
    parts.append("")
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    modeling.setup_logging()
    try:
        results = {task: evaluate_task(task) for task in TASKS}
    except (FileNotFoundError, RuntimeError) as err:
        log.error("%s", err)
        sys.exit(1)
    for task, result in results.items():
        save_metrics(result)
        log.info("%s test: accuracy %.4f | macro P %.4f R %.4f F1 %.4f | weighted F1 %.4f", task,
                 result["accuracy"], result["macro_precision"], result["macro_recall"],
                 result["macro_f1"], result["weighted_f1"])
    save_figures(results)
    comparison_table(results).to_csv(modeling.METRICS_DIR / "model_comparison.csv")
    write_report(results)
    log.info("wrote metrics, figures and %s", REPORT_PATH.relative_to(eda.ROOT))


if __name__ == "__main__":
    main()
