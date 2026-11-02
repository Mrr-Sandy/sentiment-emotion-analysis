# Sentiment and Emotion Analysis Using Machine Learning

An MCA project that classifies English short texts in two independent tasks:

| Task | Classes | Dataset |
|---|---|---|
| **A. Sentiment analysis** | negative, neutral, positive | [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`sentiment`) |
| **B. Emotion detection** | sadness, joy, love, anger, fear, surprise | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |

The planned approach is supervised learning with **TF-IDF** features and **Logistic Regression**
in **scikit-learn**.

**Status:** Phase 1 (environment setup, data collection and exploratory data analysis) is complete.
Model training starts in Phase 2.

## Project structure

```
sentiment-emotion-analysis/
├── data/
│   ├── raw/                    # downloaded CSVs (git-ignored, created by src/download_data.py)
│   ├── processed/              # cleaned data for modelling (Phase 2)
│   └── README.md               # dataset details, licenses, citations
├── notebooks/
│   ├── sentiment_eda.ipynb     # EDA for Task A
│   └── emotion_eda.ipynb       # EDA for Task B
├── reports/
│   ├── dataset_summary.md      # generated summary of both datasets
│   └── figures/                # charts saved by the notebooks
├── src/
│   ├── download_data.py        # downloads both datasets at pinned revisions
│   ├── eda.py                  # shared EDA helpers and chart style
│   └── make_dataset_summary.py # writes reports/dataset_summary.md
├── models/                     # trained models (Phase 2)
├── requirements.txt
└── README.md
```

## Setup (Windows, PowerShell)

Requires Python 3.12 (tested with 3.12.10).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation fails with *"running scripts is disabled on this system"*, allow scripts for the
current PowerShell window only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Alternatively, skip activation and call the environment's Python directly, e.g.
`.\venv\Scripts\python.exe src\download_data.py`.

In VS Code, choose `venv\Scripts\python.exe` as the interpreter (*Python: Select Interpreter*) and as
the notebook kernel.

## Reproducing Phase 1

```powershell
# 1. Download both datasets into data/raw/
python src/download_data.py

# 2. Run the EDA notebooks (or open them in VS Code and choose Run All)
jupyter nbconvert --to notebook --execute --inplace notebooks/sentiment_eda.ipynb notebooks/emotion_eda.ipynb

# 3. Regenerate the dataset summary
python src/make_dataset_summary.py
```

The notebooks save their charts to `reports/figures/`.

## Phase 1 findings

The full results are in [`reports/dataset_summary.md`](reports/dataset_summary.md).

- **Both datasets are imbalanced:** 2.9 : 1 for sentiment (neutral vs negative) and 9.4 : 1 for
  emotion (joy vs surprise). Report macro-averaged metrics, not accuracy alone.
- **The sentiment test split has a different class mix:** negative tweets make up 32% of test but
  only 16% of train.
- **Sentiment text needs cleaning:** about 11% of train and validation tweets contain literal
  escape sequences that never occur in test, plus HTML entities and `@user` placeholders.
  The emotion text is already lowercased and stripped of punctuation.
- **Emotion labels track specific keywords** (e.g. *apprehensive* is about 400 times more common in
  `fear` than elsewhere), so a bag-of-words model should do well on Task B. Sentiment's `neutral`
  class has no strong cue words and is likely to be the hardest class.
- **Keep negations:** scikit-learn's built-in English stop-word list removes *not*, *no* and *never*,
  so it should not be used for sentiment.

## Data licenses

TweetEval's sentiment data is CC BY 3.0 and subject to the Twitter/X Terms of Service; the emotion
dataset is for educational and research use only. See [`data/README.md`](data/README.md) for
details and citations.
