import numpy as np
import pandas as pd
from pathlib import Path
from typing import Final

from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"
TRAINIGN_DATA_PATH: Final[Path] = BASE_DIR / "dataset" / "training-data.npz"

if __name__ == "__main__":
    x = pd.read_csv(FEATURE_PATH)
    y = pd.read_csv(LABEL_PATH)
    logger.info(f"Features and Label dataset fetched - {x.shape}, {y.shape}")

    y = y.loc[y.ssd<"2025-01-01"].copy()
    logger.info(f"Training data duration - {y.ssd.min()} - {y.ssd.max()}")

    x_train, y_train = [], []
    for ssd in y.ssd.unique():
        temp_x = (
            x.loc[x.ssd <= ssd]
            .reset_index(drop=True)
            .iloc[-10:]
            .drop(columns=["ssd"])
            .to_numpy()
        )
        temp_y = (
            y.loc[y.ssd == ssd]
            .sort_values("days_ahead")["y"]
            .to_numpy()
        )
        if temp_x.shape[0] == 10:
            x_train.append(temp_x)
            y_train.append(temp_y)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    logger.info(f"Training data prepared - {x_train.shape}, {y_train.shape}")

    logger.info(f"Saving training dataset to {TRAINIGN_DATA_PATH}")
    np.savez_compressed(TRAINIGN_DATA_PATH, X=x_train,y=y_train)
