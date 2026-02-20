import sqlite3
from pathlib import Path
from typing import Final


from core.logging import LoggerFactory
from core.utils import FetchFromKaggle, PrepareLabels


logger = LoggerFactory().get_logger(__name__)


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATASET_PATH: Final[Path] = BASE_DIR / "dataset" / "autonomous-metal-db.db"
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
HORIZON_DAYS: Final[int] = 5


if __name__ == "__main__":
    logger.info("Starting label preparation pipeline")

    logger.info("Ensuring dataset availability via Kaggle fetch")
    FetchFromKaggle().download()

    logger.info("Connecting to database: %s", DATASET_PATH)

    with sqlite3.connect(DATASET_PATH) as conn:
        prepare_labels = PrepareLabels(conn=conn)

    label_df = prepare_labels.build_labels(HORIZON_DAYS)

    logger.info(f"Saving labels to dataset {LABEL_PATH}")
    label_df.to_csv(LABEL_PATH, index=False)
