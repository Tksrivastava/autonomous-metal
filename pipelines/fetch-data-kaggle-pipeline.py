from core.logging import LoggerFactory
from core.utils import FetchFromKaggle

logger = LoggerFactory().get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting label preparation pipeline")

    logger.info("Ensuring dataset availability via Kaggle fetch")
    FetchFromKaggle().download()
