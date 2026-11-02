# Data

The datasets are **not committed to Git**. `src/download_data.py` downloads them from the
Hugging Face Hub at pinned revisions, so every download produces exactly the same files.

## Download

```powershell
python src/download_data.py                    # both datasets
python src/download_data.py --dataset emotion  # one dataset
python src/download_data.py --force            # re-download and overwrite
```

## Layout

```
data/
├── raw/
│   ├── sentiment/   train.csv, validation.csv, test.csv
│   └── emotion/     train.csv, validation.csv, test.csv
└── processed/       cleaned data for modelling (Phase 2)
```

Every CSV has three columns: `text`, `label` (integer id) and `label_name`.

## Task A: TweetEval sentiment

| | |
|---|---|
| Hugging Face | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), config `sentiment` |
| Revision | `b3a375baf0f409c77e6bc7aa35102b7b3534f8be` |
| Labels | 0 `negative`, 1 `neutral`, 2 `positive` |
| Rows | train 45,615 · validation 2,000 · test 12,284 |
| Origin | English tweets from SemEval-2017 Task 4 (subtask A); user names replaced with `@user` |
| License | CC BY 3.0 (SemEval-2017 data); use must also comply with the Twitter/X Terms of Service |

## Task B: Emotion

| | |
|---|---|
| Hugging Face | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), config `split` |
| Revision | `cab853a1dbdf4c42c2b3ef2173804746df8825fe` |
| Labels | 0 `sadness`, 1 `joy`, 2 `love`, 3 `anger`, 4 `fear`, 5 `surprise` |
| Rows | train 16,000 · validation 2,000 · test 2,000 |
| Origin | English Twitter messages from the CARER paper, distributed already lowercased and without punctuation |
| License | Educational and research use only (per the dataset card) |

## Known issues

Found during the Phase 1 EDA; details in [`reports/dataset_summary.md`](../reports/dataset_summary.md).

- **Sentiment:** about 11% of train and validation texts contain literal escape sequences
  (a backslash, `u` and four hex digits, e.g. for `’` and `,`), but no test text does. Decode them
  before modelling (`decode_artifacts()` in `src/eda.py`). HTML entities such as `&amp;` also occur.
- **Sentiment:** the test split's class mix differs strongly from train (negative 32% vs 16%).
- **Both:** a few identical texts carry different labels (3 in sentiment train, 30 in emotion train).
- **Emotion:** 19 distinct texts appear in more than one split (11 in both train and test).

## Citations

- Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L. and Neves, L. (2020). *TweetEval: Unified
  Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP 2020.
- Rosenthal, S., Farra, N. and Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in
  Twitter.* Proceedings of SemEval-2017, pp. 502-518.
- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J. and Chen, Y.-S. (2018). *CARER: Contextualized
  Affect Representations for Emotion Recognition.* Proceedings of EMNLP 2018, pp. 3687-3697.
