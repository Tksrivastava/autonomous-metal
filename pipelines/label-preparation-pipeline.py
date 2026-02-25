import os
import sqlite3
from pathlib import Path
from typing import Final
from dotenv import load_dotenv
from core.utils import PrepareLabels
from core.logging import LoggerFactory


logger = LoggerFactory().get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATASET_PATH: Final[Path] = BASE_DIR / "dataset" / "autonomous-metal-db.db"
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
HORIZON_DAYS: Final[int] = int(os.getenv("FORECAST_HORIZON"))


if __name__ == "__main__":
    logger.info("Starting label preparation pipeline")

    logger.info("Connecting to database: %s", DATASET_PATH)

    with sqlite3.connect(DATASET_PATH) as conn:
        prepare_labels = PrepareLabels(conn=conn)

    label_df = prepare_labels.build_labels(HORIZON_DAYS)

    logger.info(f"Saving labels to dataset {LABEL_PATH}")
    label_df.to_csv(LABEL_PATH, index=False)
