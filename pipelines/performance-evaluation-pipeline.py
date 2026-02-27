import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Final
from dotenv import load_dotenv
from core.logging import LoggerFactory
from tensorflow.keras.backend import clear_session
from core.model import AutonomousForecastModelArchitecture

logger = LoggerFactory().get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
LABEL_PATH: Final[Path] = BASE_DIR / "dataset" / "labels.csv"
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"
SCALER_MODEL_PATH: Final[Path] = BASE_DIR / "artifacts" / "feature-scaler.pkl"
MODEL_PATH: Final[Path] = (
    BASE_DIR / "artifacts" / "lme-al-forecast-model-%s-days-ahead.keras"
)
HORIZON_DAYS: Final[int] = int(os.getenv("FORECAST_HORIZON"))
LAG_WINDOW: int = int(os.getenv("LAG_WINDOW"))


if __name__ == "__main__":
    logger.info("Starting forecast performance evaluation pipeline")

    logger.info(f"Loading scaler from - {SCALER_MODEL_PATH}")
    scaler = pickle.load(open(SCALER_MODEL_PATH, "rb"))

    logger.info("Loading data")
    x = pd.read_csv(FEATURE_PATH).sort_values("ssd").reset_index(drop=True)
    y = (
        pd.read_csv(LABEL_PATH)
        .sort_values(["ssd", "days_ahead"])
        .reset_index(drop=True)
    )

    feature_cols = [c for c in x.columns if c != "ssd"]

    X_values = x[feature_cols].to_numpy()
    ssd_values = x["ssd"].to_numpy()

    logger.info("Building sliding windows")

    windows = []
    window_ssd = []

    for i in range(LAG_WINDOW - 1, len(X_values)):
        windows.append(X_values[i - LAG_WINDOW + 1 : i + 1])
        window_ssd.append(ssd_values[i])

    windows = np.array(windows)  # (N, WINDOW, n_features)

    logger.info(f"Created windows: {windows.shape}")

    n_samples, n_steps, n_features = windows.shape

    windows_2d = windows.reshape(-1, n_features)
    windows_scaled = scaler.transform(windows_2d)
    windows_scaled = windows_scaled.reshape(n_samples, n_steps, n_features)

    logger.info("Running batch prediction for each horizon")
    preds = []
    for days_ahead in range(HORIZON_DAYS):
        logger.info(f"Loading model from - {str(MODEL_PATH)%str(days_ahead+1)}")
        model = AutonomousForecastModelArchitecture.load(
            str(MODEL_PATH) % str(days_ahead + 1)
        )
        logger.info("Model loaded")
        preds.append(model.predict(windows_scaled))

        clear_session()
        del model
    preds = np.hstack(preds)

    horizon = preds.shape[1]

    logger.info("Aligning predictions with (ssd, days_ahead)")

    rows = []
    for ssd_val, pred_vec in zip(window_ssd, preds):
        for h, pred in enumerate(pred_vec, start=1):
            rows.append((ssd_val, h, float(pred)))

    pred_df = pd.DataFrame(
        rows,
        columns=["ssd", "days_ahead", "prediction"],
    )

    # merge predictions into labels
    y = y.merge(pred_df, on=["ssd", "days_ahead"], how="left")

    y = y.loc[(~y.prediction.isna()) & (~y.y.isna())].copy()

    logger.info(f"Predictions generated - {y.shape}")

    logger.info("Evaluating performance")

    y["actual_price"] = (y["current_spot_price"] * y["y"]) + y["current_spot_price"]

    y["predicted_price"] = (y["current_spot_price"] * y["prediction"]) + y[
        "current_spot_price"
    ]

    y["ape"] = abs(y["predicted_price"] - y["actual_price"]) / y["actual_price"]

    y["da"] = np.where(np.sign(y["y"]) == np.sign(y["prediction"]), 1, 0)

    y["period"] = np.where(
        y["ssd"] >= "2025-02-01",
        "validation",
        "train",
    )

    logger.info("MAPE ->")
    logger.info(
        y.pivot_table(
            index="days_ahead",
            columns="period",
            values="ape",
            aggfunc=["mean", "count"],
        ).reset_index()
    )

    logger.info("Directional Accuracy ->")
    logger.info(
        y.pivot_table(
            index="days_ahead",
            columns="period",
            values="da",
            aggfunc=["mean", "count"],
        ).reset_index()
    )
