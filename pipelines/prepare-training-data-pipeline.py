import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Final
from dotenv import load_dotenv
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"
ARTIFACT_PATH: Final[Path] = BASE_DIR / "artifacts"
TRAINING_X_PATH: Final[Path] = BASE_DIR / "artifacts" / "training-x.pkl"
TRAINING_Y_PATH: Final[Path] = BASE_DIR / "artifacts" / "training-y.pkl"
FEATURES_SET_PATH: Final[Path] = BASE_DIR / "artifacts" / "features-set.pkl"
SPOT_PRICES_PATH: Final[Path] = BASE_DIR / "artifacts" / "spot-prices.csv"
LAG_WINDOW: int = int(os.getenv("LAG_WINDOW"))

if __name__ == "__main__":
    x = pd.read_csv(FEATURE_PATH)
    x = x.sort_values("ssd").reset_index(drop=True)
    y = pd.read_csv(LABEL_PATH)
    logger.info(f"Features and Label dataset fetched - {x.shape}, {y.shape}")

    features_set = x.drop(columns=["ssd"]).columns.tolist()
    pickle.dump(features_set, open(FEATURES_SET_PATH, "wb"))
    logger.info(f"Features order saved - {len(features_set)}")

    actual = y[["ssd", "current_spot_price"]].copy()
    actual.to_csv(SPOT_PRICES_PATH, index=False)
    logger.info(f"Spot prices saved - {actual.shape}")

    y = (
        y.loc[y.ssd < "2025-02-01"]
        .sort_values("ssd", ascending=True)
        .reset_index(drop=True)
    )
    logger.info(f"Training data duration - {y.ssd.min()} - {y.ssd.max()}")

    x_train, y_train = {}, {}
    for ssd in y.ssd.unique():
        temp_x = (
            x.loc[x.ssd <= ssd][features_set]
            .reset_index(drop=True)
            .iloc[-LAG_WINDOW:]
            .drop(columns=["ssd"])
            .to_numpy()
        )
        temp_y = y.loc[(y.ssd == ssd)].sort_values("days_ahead")["y"].to_numpy()
        if temp_x.shape[0] == 10:
            x_train[ssd] = temp_x
            y_train[ssd] = temp_y
    logger.info(f"Training data prepared - {len(x_train)}, {len(y_train)}")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact directory ready: %s", ARTIFACT_PATH)

    logger.info(f"Saving training dataset to {TRAINING_X_PATH} and {TRAINING_Y_PATH}")
    pickle.dump(x_train, open(TRAINING_X_PATH, "wb"))
    pickle.dump(y_train, open(TRAINING_Y_PATH, "wb"))
