"""Train the Task B emotion model (sadness / joy / love / anger / fear / surprise).

    python src/train_emotion.py

Run src/prepare_data.py first. Saves models/emotion/{vectorizer,model}.joblib
and metadata.json, and the validation grid to reports/metrics/.
"""
import logging
import sys

import modeling


def main() -> None:
    modeling.setup_logging()
    try:
        modeling.train("emotion")
    except FileNotFoundError as err:
        logging.getLogger("train_emotion").error("%s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
