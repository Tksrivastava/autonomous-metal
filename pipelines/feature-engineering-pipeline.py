import sqlite3
from pathlib import Path
from typing import Final


from core.logging import LoggerFactory
from core.utils import FetchRawFeatures


logger = LoggerFactory().get_logger(__name__)


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATASET_PATH: Final[Path] = BASE_DIR / "dataset" / "autonomous-metal-db.db"
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"

if __name__ == "__main__":
    logger.info("Starting label preparation pipeline")

    logger.info("Connecting to database: %s", DATASET_PATH)

    with sqlite3.connect(DATASET_PATH) as conn:
        raw_features = FetchRawFeatures(conn=conn)

    features = raw_features.fetch()

    logger.info(f"Saving raw features to dataset to {FEATURE_PATH}")
    features.to_csv(FEATURE_PATH, index=False)
