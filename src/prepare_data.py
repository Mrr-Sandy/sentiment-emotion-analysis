"""Prepare the Phase 1 raw data for modelling.

    python src/prepare_data.py

Reads data/raw/<task>/{train,validation,test}.csv (never modified) and writes
data/processed/<task>/{train,validation,test}.csv plus prep_summary.json.
Only the training split is cleaned: texts that carry conflicting labels are
removed and exact duplicates are kept once. Validation and test are copied
unchanged so results stay comparable with published numbers.
"""
import json
import logging
import sys

import pandas as pd

import eda

PROCESSED_DIR = eda.ROOT / "data" / "processed"
TASKS = ("sentiment", "emotion")
REQUIRED_COLUMNS = ["text", "label", "label_name"]

log = logging.getLogger("prepare_data")


def check_split(task: str, split: str, df: pd.DataFrame, labels: list[str]) -> dict:
    """Validate one split's columns and labels; return its quality counts."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{task}/{split}.csv is missing columns {missing_columns}")
    unknown = sorted(set(df["label_name"]) - set(labels))
    if unknown:
        raise ValueError(f"{task}/{split}.csv has unexpected labels {unknown}")
    pairs = df[["label", "label_name"]].drop_duplicates()
    if len(pairs) != df["label"].nunique():
        raise ValueError(f"{task}/{split}.csv maps a label id to more than one class name")
    return {
        "rows": len(df),
        "empty_texts": int((df["text"].str.strip() == "").sum()),
        "repeated_texts": int(df["text"].duplicated().sum()),
        "conflicting_label_texts": int(df.groupby("text")["label"].nunique().gt(1).sum()),
    }


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    """Drop empty texts and texts with conflicting labels; keep one copy of exact duplicates."""
    conflicted = train.groupby("text")["label"].transform("nunique") > 1
    empty = train["text"].str.strip() == ""
    return train[~conflicted & ~empty].drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def prepare(task: str) -> dict:
    splits = eda.load_splits(task)
    labels = eda.label_order(splits)
    summary = {
        "task": task,
        "labels": labels,
        "raw": {split: check_split(task, split, df, labels) for split, df in splits.items()},
        "shared_texts_between_splits": eda.split_overlap(splits)["shared_texts"].to_dict(),
    }

    prepared = dict(splits, train=clean_train(splits["train"]))
    out_dir = PROCESSED_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["processed_rows"] = {}
    for split, df in prepared.items():
        df.to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
        summary["processed_rows"][split] = len(df)
    summary["train_rows_removed"] = len(splits["train"]) - len(prepared["train"])

    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%s: train %d -> %d rows (%d removed); validation %d and test %d unchanged",
             task, len(splits["train"]), len(prepared["train"]), summary["train_rows_removed"],
             len(prepared["validation"]), len(prepared["test"]))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        for task in TASKS:
            prepare(task)
    except (FileNotFoundError, ValueError) as err:
        log.error("%s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
