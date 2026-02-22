from pathlib import Path
from typing import Final
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"
SCALER_MODEL_PATH: Final[Path] = BASE_DIR / "artifcats" / "feature-scaler.pkl"
FORECAST_MODEL_PATH: Final[Path] = (
    BASE_DIR / "artifcats" / "lme-al-forecast-model.keras"
)
