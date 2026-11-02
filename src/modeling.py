"""Shared training logic for both tasks: TF-IDF + Logistic Regression.

For each task, every TF-IDF configuration is fitted on the training split only,
every Logistic Regression setting is trained on it, and all candidates are
scored on the validation split. The candidate with the best validation macro F1
is refitted on the training split and saved to models/<task>/. The test split
is not read here; it is used once, by src/evaluate.py.
"""
import json
import logging
import platform
import time
from datetime import datetime, timezone
from itertools import product

import joblib
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import eda
import preprocessing
from download_data import DATASETS
from prepare_data import PROCESSED_DIR

SEED = 42
MAX_ITER = 3000
MODELS_DIR = eda.ROOT / "models"
METRICS_DIR = eda.ROOT / "reports" / "metrics"

TFIDF_CONFIGS = {
    "unigrams": {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": None},
    "uni+bigrams, 20k": {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 20_000},
    "uni+bigrams, 50k": {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 50_000},
    "uni+bigrams, all": {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": None},
}
C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CLASS_WEIGHTS = [None, "balanced"]

log = logging.getLogger("modeling")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def load_processed(task: str, split: str) -> pd.DataFrame:
    """Load data/processed/<task>/<split>.csv written by src/prepare_data.py."""
    path = PROCESSED_DIR / task / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/prepare_data.py")
    return pd.read_csv(path, keep_default_na=False)


def make_model(C: float, class_weight) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=MAX_ITER, random_state=SEED)


def _fit_and_score(X_train, y_train, X_val, y_val, C, class_weight) -> dict:
    start = time.perf_counter()
    model = make_model(C, class_weight).fit(X_train, y_train)
    predicted = model.predict(X_val)
    return {
        "C": C,
        "class_weight": class_weight or "none",
        "val_macro_f1": f1_score(y_val, predicted, average="macro"),
        "val_weighted_f1": f1_score(y_val, predicted, average="weighted"),
        "val_accuracy": accuracy_score(y_val, predicted),
        "converged": bool(model.n_iter_.max() < MAX_ITER),
        "fit_seconds": round(time.perf_counter() - start, 1),
    }


def search(task: str, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Score every TF-IDF x Logistic Regression candidate on the validation split."""
    rows = []
    for name, config in TFIDF_CONFIGS.items():
        vectorizer = preprocessing.build_vectorizer(**config)
        X_train = vectorizer.fit_transform(train["text"])  # fitted on training data only
        X_val = vectorizer.transform(validation["text"])
        scores = Parallel(n_jobs=-1)(
            delayed(_fit_and_score)(X_train, train["label_name"], X_val, validation["label_name"], C, cw)
            for C, cw in product(C_VALUES, CLASS_WEIGHTS)
        )
        best = max(s["val_macro_f1"] for s in scores)
        log.info("%s | TF-IDF %-17s %6d features | best validation macro F1 %.4f",
                 task, name, len(vectorizer.vocabulary_), best)
        rows += [{"tfidf": name, "n_features": len(vectorizer.vocabulary_), **s} for s in scores]
    table = pd.DataFrame(rows)
    if not table["converged"].all():
        log.warning("%s: %d candidates hit max_iter=%d", task, (~table["converged"]).sum(), MAX_ITER)
    # Best macro F1 first; ties go to higher accuracy, then the smaller vocabulary.
    return table.sort_values(["val_macro_f1", "val_accuracy", "n_features"],
                             ascending=[False, False, True]).reset_index(drop=True)


def train(task: str) -> dict:
    """Select, refit and save the task's model; return its metadata."""
    train_df, val_df = load_processed(task, "train"), load_processed(task, "validation")
    labels = json.loads((PROCESSED_DIR / task / "prep_summary.json").read_text(encoding="utf-8"))["labels"]
    log.info("%s: %d training and %d validation texts, classes %s", task, len(train_df), len(val_df), labels)

    table = search(task, train_df, val_df)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / f"{task}_validation_results.csv", index=False)
    best = table.iloc[0]
    tfidf_config = TFIDF_CONFIGS[best["tfidf"]]
    class_weight = None if best["class_weight"] == "none" else best["class_weight"]
    log.info("%s: selected TF-IDF '%s', C=%s, class_weight=%s (validation macro F1 %.4f)",
             task, best["tfidf"], best["C"], best["class_weight"], best["val_macro_f1"])

    vectorizer = preprocessing.build_vectorizer(**tfidf_config)
    X_train = vectorizer.fit_transform(train_df["text"])
    model = make_model(float(best["C"]), class_weight).fit(X_train, train_df["label_name"])
    # stop_words_ only lists the terms pruned by min_df/max_df/max_features; it is
    # not needed for prediction and would bloat the saved file.
    vectorizer.stop_words_ = None

    best_by_weight = table.groupby("class_weight")["val_macro_f1"].max().round(4).to_dict()
    metadata = {
        "task": task,
        "labels": labels,
        "model_classes": model.classes_.tolist(),
        "tfidf": {"name": best["tfidf"], **tfidf_config, "n_features": len(vectorizer.vocabulary_),
                  "sublinear_tf": True, "token_pattern": preprocessing.TOKEN_PATTERN},
        "logistic_regression": {"C": float(best["C"]), "class_weight": class_weight,
                                "max_iter": MAX_ITER, "solver": model.solver, "random_state": SEED},
        "preprocessing": {"function": "preprocessing.clean_text",
                          "version": preprocessing.PREPROCESSING_VERSION},
        "validation": {"macro_f1": round(float(best["val_macro_f1"]), 4),
                       "weighted_f1": round(float(best["val_weighted_f1"]), 4),
                       "accuracy": round(float(best["val_accuracy"]), 4),
                       "best_macro_f1_by_class_weight": best_by_weight,
                       "candidates": len(table)},
        "data": {"dataset": DATASETS[task]["path"], "config": DATASETS[task]["config"],
                 "revision": DATASETS[task]["revision"], "train_rows": len(train_df)},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir = MODELS_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib", compress=3)
    joblib.dump(model, out_dir / "model.joblib", compress=3)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("%s: saved vectorizer, model and metadata to %s", task, out_dir.relative_to(eda.ROOT))
    return metadata
