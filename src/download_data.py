"""Download the two Phase 1 datasets from Hugging Face into data/raw/.

    python src/download_data.py                      # both datasets
    python src/download_data.py --dataset emotion    # one dataset
    python src/download_data.py --force              # re-download and overwrite

Each split is saved as data/raw/<dataset>/<split>.csv with the columns
text, label (integer id) and label_name (string).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Revisions are pinned to the Hub commits used for this project, so every
# download yields exactly the same data.
DATASETS = {
    "sentiment": {
        "path": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
    },
    "emotion": {
        "path": "dair-ai/emotion",
        "config": "split",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
    },
}


def download(key: str, force: bool = False) -> None:
    spec = DATASETS[key]
    out_dir = RAW_DIR / key
    if not force and out_dir.exists() and any(out_dir.glob("*.csv")):
        print(f"[{key}] already downloaded in {out_dir} (use --force to refresh)")
        return

    print(f"[{key}] loading {spec['path']} (config: {spec['config']}, revision: {spec['revision'][:7]})")
    dataset = load_dataset(spec["path"], spec["config"], revision=spec["revision"])
    label_names = dataset["train"].features["label"].names
    id_to_name = dict(enumerate(label_names))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, part in dataset.items():
        df = part.to_pandas()
        df["label_name"] = df["label"].map(id_to_name)
        path = out_dir / f"{split}.csv"
        df.to_csv(path, index=False, encoding="utf-8")

        # Verify the file on disk matches what was downloaded.
        check = pd.read_csv(path)
        assert len(check) == len(part), f"{path}: row count changed on write"
        assert check["label_name"].notna().all(), f"{path}: unmapped labels"
        print(f"[{key}] {split:<10} {len(check):>7,} rows -> {path.relative_to(ROOT)}")

    print(f"[{key}] labels: {id_to_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download and overwrite")
    args = parser.parse_args()

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        download(key, force=args.force)


if __name__ == "__main__":
    main()
