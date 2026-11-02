"""Cached access to the saved models and the Phase 1 / Phase 2 reports.

Only files committed to Git are read (models/ and reports/), so the app also
runs where data/raw/ is absent, e.g. on Streamlit Community Cloud.
"""
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import load_classifier  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
TASKS = ("sentiment", "emotion")

log = logging.getLogger(__name__)


class ResourceError(RuntimeError):
    """A required file could not be used; the message is safe to show to users."""


@st.cache_resource(show_spinner="Loading the trained models…")
def get_classifiers() -> dict:
    """Load both saved classifiers once per server process."""
    try:
        return {task: load_classifier(task) for task in TASKS}
    except Exception as err:  # noqa: BLE001 - details go to the server log, not the page
        log.exception("Could not load the saved models")
        raise ResourceError("The trained models could not be loaded. Make sure the models/ folder "
                            "from the repository is present.") from err


@st.cache_data(show_spinner=False)
def test_metrics(task: str) -> dict:
    """Phase 2 test-set metrics written by src/evaluate.py."""
    path = METRICS / f"{task}_test_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        log.exception("Could not read %s", path.name)
        raise ResourceError("The evaluation results are not available.") from err


CLASS_ROW_RE = re.compile(
    r"^\| (?P<label>[a-z]+) \(\d+\) \| (?P<train>[\d,]+) \([\d.]+%\) \| (?P<validation>[\d,]+) \([\d.]+%\) "
    r"\| (?P<test>[\d,]+) \([\d.]+%\) \|$", re.MULTILINE)


@st.cache_data(show_spinner=False)
def class_counts() -> dict[str, pd.DataFrame]:
    """Examples per class and split, read from the Phase 1 reports/dataset_summary.md."""
    try:
        text = (REPORTS / "dataset_summary.md").read_text(encoding="utf-8")
        sections = {
            "sentiment": text.split("\n## Task A")[1].split("\n## Task B")[0],
            "emotion": text.split("\n## Task B")[1].split("\n## Implications")[0],
        }
    except (OSError, IndexError) as err:
        log.exception("Could not read the dataset summary")
        raise ResourceError("The dataset summary is not available.") from err

    counts = {}
    for task, section in sections.items():
        rows = [m.groupdict() for m in CLASS_ROW_RE.finditer(section)]
        if not rows:
            raise ResourceError("The dataset summary has an unexpected format.")
        frame = pd.DataFrame(rows).set_index("label")
        counts[task] = frame.apply(lambda col: col.str.replace(",", "").astype(int))
    return counts


def figure(name: str) -> Path:
    """Path of reports/figures/<name>.png."""
    path = FIGURES / f"{name}.png"
    if not path.exists():
        raise ResourceError("A report figure is missing.")
    return path
