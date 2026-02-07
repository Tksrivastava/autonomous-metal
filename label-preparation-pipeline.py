import sqlite3
import pandas as pd
from pathlib import Path

from core.logging import LoggerFactory
from core.utils import FetchFromKaggle


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


FILE_PATH = Path(__file__).resolve().parents[0]
DATASET_PATH = FILE_PATH / "dataset" / "autonomous-metal-db.db"

def main() -> None:
    """
    Execute the feature generation pipeline.
    """
    logger.info("Starting label preparation pipeline")

    logger.info("Ensuring dataset availability via Kaggle fetch")
    FetchFromKaggle().download()

    logger.info(f"Establishing connection to database from {DATASET_PATH}")
    conn = sqlite3.connect(DATASET_PATH)

    logger.info("Fetching LME Aluminum spot price")
    df = pd.read_sql("SELECT * FROM `lme-aluminum-spot-prices`", conn)
    logger.info(f"Data fetched - {df.shape}")

if __name__ == "__main__":
    main()