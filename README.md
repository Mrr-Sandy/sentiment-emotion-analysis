# Sentiment and Emotion Analysis Using Machine Learning

An MCA project that classifies English short texts in two independent tasks:

| Task | Classes | Dataset |
|---|---|---|
| **A. Sentiment analysis** | negative, neutral, positive | [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`sentiment`) |
| **B. Emotion detection** | sadness, joy, love, anger, fear, surprise | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |

The planned approach is supervised learning with **TF-IDF** features and **Logistic Regression**
in **scikit-learn**.

**Status:** Phase 1 (environment setup, data collection and exploratory data analysis),
Phase 2 (model training and evaluation) and Phase 3 (the **Sentiment & Emotion Intelligence**
Streamlit web app) are complete. Both trained models are saved in `models/`.

## Web app: Sentiment & Emotion Intelligence

*Machine Learning Based Text Sentiment and Emotion Analysis.*

```powershell
streamlit run app.py
```

Then open http://localhost:8501. The app loads the saved models once (cached) and never retrains.

| Page | What it shows |
|---|---|
| **Home** | Project overview, both tasks, the model pipeline, dataset sizes, headline test results, technology stack |
| **Text Analyzer** | Type or pick an example sentence; both models classify it independently and show the predicted sentiment and emotion, model confidence and the full probability distribution for every class |
| **Model Performance** | Phase 2 test metrics (accuracy, macro / weighted precision, recall, F1), comparison chart, confusion matrices, per-class scores, and what each metric means |
| **Dataset Analysis** | Split sizes, examples per class, key findings and the Phase 1 EDA charts, with sources and citations |

Other features: sidebar navigation, light and dark themes (⋮ → Settings), layouts that adapt to
phone screens, input validation (empty, too long, no words, words the models have never seen), and
error messages that never expose file paths or stack traces.

**Model architecture:** text → preprocessing (`src/preprocessing.py`, built into each saved
vectorizer) → TF-IDF → Logistic Regression → class probabilities, run separately for sentiment and
emotion. The app calls `load_classifier()` from `src/predict.py`; it has no preprocessing or
vectorizer code of its own.

### Screenshots

<!-- Add screenshots here, e.g. docs/screenshots/home.png, analyzer.png, performance.png -->
_Screenshots to be added._

### Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (the app needs `app.py`, `webapp/`, `src/`, `models/`, `reports/`,
   `.streamlit/config.toml` and `requirements.txt`; the raw data is not needed).
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**, select this
   repository and branch `main`, and set the main file path to `app.py`.
3. Under **Advanced settings**, choose Python **3.12** so the pinned packages (and the scikit-learn
   version the models were saved with, 1.9.1) install as tested.
4. Deploy. No secrets or API keys are required.

The app code is in `app.py` (entry point and navigation) and `webapp/`: `inference.py` (input
validation and prediction, no UI code), `resources.py` (cached loading of models and reports),
`ui.py` (styling and charts) and `views/` (one file per page).

## Project structure

```
sentiment-emotion-analysis/
├── data/
│   ├── raw/                    # downloaded CSVs (git-ignored, created by src/download_data.py)
│   ├── processed/              # modelling data (git-ignored, created by src/prepare_data.py)
│   └── README.md               # dataset details, licenses, citations
├── notebooks/
│   ├── sentiment_eda.ipynb     # EDA for Task A
│   └── emotion_eda.ipynb       # EDA for Task B
├── reports/
│   ├── dataset_summary.md      # generated summary of both datasets
│   ├── model_evaluation.md     # generated evaluation report (Phase 2)
│   ├── metrics/                # test metrics, classification reports, validation grids
│   └── figures/                # charts from the notebooks and the evaluation
├── src/
│   ├── download_data.py        # downloads both datasets at pinned revisions
│   ├── eda.py                  # shared EDA helpers and chart style
│   ├── make_dataset_summary.py # writes reports/dataset_summary.md
│   ├── preprocessing.py        # text cleaning + TF-IDF builder shared by training and inference
│   ├── prepare_data.py         # builds data/processed/ from data/raw/
│   ├── modeling.py             # model selection on validation, training, saving
│   ├── train_sentiment.py      # trains the Task A model
│   ├── train_emotion.py        # trains the Task B model
│   ├── evaluate.py             # test-set evaluation, metrics, figures, report
│   └── predict.py              # loads saved models and classifies new text
├── models/
│   ├── sentiment/              # vectorizer.joblib, model.joblib, metadata.json
│   └── emotion/                # vectorizer.joblib, model.joblib, metadata.json
├── webapp/
│   ├── inference.py            # input validation and prediction for the app (no UI code)
│   ├── resources.py            # cached loading of models, metrics and report data
│   ├── ui.py                   # shared styling, header, pipeline diagram and charts
│   └── views/                  # home, analyzer, performance and dataset pages
├── .streamlit/config.toml      # light/dark themes, hides error details from visitors
├── app.py                      # Streamlit entry point
├── tests/                      # unit and app tests (python -m unittest)
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

## Phase 2: models

### Algorithms

Each task has its own independent pipeline: **TF-IDF** features followed by a multinomial
**Logistic Regression** classifier (scikit-learn, `lbfgs` solver).

- **TF-IDF** (term frequency × inverse document frequency) turns each text into a sparse vector
  with one weight per word or word pair (n-gram). A term's weight grows with how often it occurs in
  the text (here sublinearly, `1 + log(tf)`) and shrinks with how many training texts contain it, so
  words that appear everywhere (such as "feel" in the emotion data) count for little, and rarer,
  more telling words count for more.
- **Logistic Regression** learns one weight per feature and class and turns the weighted sum into
  class probabilities with the softmax function. It is fast, works well with sparse TF-IDF
  vectors, and its weights are easy to inspect. `C` controls regularisation strength, and
  `class_weight="balanced"` gives rare classes more weight during training.

### Preprocessing

`src/preprocessing.py` defines `clean_text()`, which is built into every saved vectorizer, so
training and prediction always clean text the same way. It decodes escape sequences and HTML
entities (Phase 1's `decode_artifacts`), lowercases, replaces URLs and @mentions with placeholder
words, strips `#` from hashtags and spells negations out as `not` (`don't`, `dont` → `do not`). No
stop-word list is used, so *not*, *no* and *never* survive, and `!`, `?`, emoticons and emoji are
kept as tokens.

### Training and evaluation

1. `src/prepare_data.py` copies the Phase 1 splits to `data/processed/`. In the training split it
   removes texts that carry conflicting labels and keeps one copy of exact duplicates (32 sentiment
   and 61 emotion rows). Validation and test are used unchanged; `data/raw/` is never modified.
2. `src/train_sentiment.py` and `src/train_emotion.py` try 4 TF-IDF configurations (unigrams;
   unigrams + bigrams capped at 20k, at 50k, or uncapped; all with `min_df=2`, `max_df=0.95`) × 7
   values of `C` (0.25-16) × `class_weight` None or `"balanced"`: 56 candidates per task. The
   vectorizer is fitted on the training split only. The candidate with the best **validation macro
   F1** is refitted on the training split and saved. Random seeds are fixed (`random_state=42`).
3. `src/evaluate.py` runs the saved models once on the untouched test splits and writes accuracy,
   per-class and macro/weighted precision, recall and F1, classification reports, confusion
   matrices and charts to `reports/`, plus [`reports/model_evaluation.md`](reports/model_evaluation.md).

```powershell
python src/prepare_data.py
python src/train_sentiment.py
python src/train_emotion.py
python src/evaluate.py
python -m unittest discover -s tests -v
```

### Results (test splits)

| Metric | Sentiment (TweetEval) | Emotion (dair-ai) |
|---|---:|---:|
| Accuracy | 59.8% | 88.4% |
| Macro precision | 58.8% | 81.9% |
| Macro recall | 60.2% | 87.1% |
| **Macro F1** | **59.4%** | **84.0%** |
| Weighted F1 | 59.8% | 88.7% |
| Selected TF-IDF | unigrams + bigrams, 95,413 features | unigrams, 7,219 features |
| Selected Logistic Regression | C = 4, balanced | C = 4, balanced |
| Validation macro F1 | 66.6% | 87.7% |

The sentiment model scores 7 points lower on test than on validation, as expected from the test
split's different class mix (Phase 1); most of its errors are between `neutral` and the two polar
classes. The emotion model's weakest classes are the smallest ones, `surprise` (F1 69.3%) and
`love` (77.7%), which it mostly confuses with `fear` and `joy`. Details, confusion matrices and
per-class scores are in [`reports/model_evaluation.md`](reports/model_evaluation.md).

### Using the saved models

```python
import sys; sys.path.insert(0, "src")
from predict import load_classifier

sentiment = load_classifier("sentiment")
sentiment.predict(["Worst customer service ever."])        # ['negative']
load_classifier("emotion").predict_proba(["i feel nervous about my exam"])
```

Each `models/<task>/` folder holds `vectorizer.joblib`, `model.joblib` and `metadata.json` (classes,
selected configuration, validation scores, dataset revision, library versions). `load_classifier`
refuses artifacts built with a different preprocessing version.

## Data licenses

TweetEval's sentiment data is CC BY 3.0 and subject to the Twitter/X Terms of Service; the emotion
dataset is for educational and research use only. See [`data/README.md`](data/README.md) for
details and citations.
