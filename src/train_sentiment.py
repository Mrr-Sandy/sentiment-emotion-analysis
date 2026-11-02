"""Train the Task A sentiment model (negative / neutral / positive).

    python src/train_sentiment.py

Run src/prepare_data.py first. Saves models/sentiment/{vectorizer,model}.joblib
and metadata.json, and the validation grid to reports/metrics/.
"""
import logging
import sys

import modeling


def main() -> None:
    modeling.setup_logging()
    try:
        modeling.train("sentiment")
    except FileNotFoundError as err:
        logging.getLogger("train_sentiment").error("%s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
