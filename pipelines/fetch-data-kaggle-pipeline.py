import os
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from core.logging import LoggerFactory
from core.utils import FetchFromKaggle

logger = LoggerFactory().get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")

if __name__ == "__main__":
    logger.info("Starting label preparation pipeline")

    logger.info("Ensuring dataset availability via Kaggle fetch")
    FetchFromKaggle(
        save_path=FILE_PATH, kaggle_dataset=os.getenv("KAGGLE_DATASET")
    ).download()
