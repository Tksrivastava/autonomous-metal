import os
import pickle
import numpy as np
from pathlib import Path
from typing import Final
from dotenv import load_dotenv
from core.utils import PlotHistory
from core.logging import LoggerFactory
from sklearn.preprocessing import RobustScaler
from core.model import AutonomusForecastModelArchitecture

logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)

FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
TRAINING_X_PATH: Final[Path] = BASE_DIR / "artifacts" / "training-x.pkl"
TRAINING_Y_PATH: Final[Path] = BASE_DIR / "artifacts" / "training-y.pkl"
SCALER_PATH: Final[Path] = BASE_DIR / "artifacts" / "feature-scaler.pkl"
MODEL_PATH: Final[Path] = (
    BASE_DIR / "artifacts" / "lme-al-forecast-model-%s-days-ahead.keras"
)
PLOT_PATH: Final[Path] = BASE_DIR / "artifacts" / "loss-plot-%s-days-ahead.png"
HORIZON_DAYS: Final[int] = int(os.getenv("FORECAST_HORIZON"))


if __name__ == "__main__":
    logger.info("Starting autoencoder training pipeline")

    logger.info("Loading x data from %s", TRAINING_X_PATH)
    x = pickle.load(open(TRAINING_X_PATH, "rb"))
    x = np.stack(list(x.values())).astype(np.float32)

    logger.info("Loading y data from %s", TRAINING_Y_PATH)
    y = pickle.load(open(TRAINING_Y_PATH, "rb"))
    y = np.stack(list(y.values())).astype(np.float32)

    logger.info("Applying RobustScaler on X data")
    scaler = RobustScaler()

    n_samples, n_steps, n_features = x.shape
    x = x.reshape(-1, n_features)  ## Converting x to 2D
    x = scaler.fit_transform(x).reshape(
        n_samples, n_steps, n_features
    )  ## Converting to 3D after scaling

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Feature scaler saved to %s", SCALER_PATH)

    logger.info("Training individual model for each future horizon")
    for days_ahead in range(HORIZON_DAYS):
        horizon_y = y[:, days_ahead].reshape(-1, 1)
        logger.info(
            f"Filtered label data for {days_ahead+1} days ahead horizon - {horizon_y.shape}"
        )

        logger.info(
            f"Initializing model | input_space={x.shape[1]} | output_space={horizon_y.shape[1]}"
        )
        model = AutonomusForecastModelArchitecture(
            seed=42,
            input_horizon_space=n_steps,
            input_feature_space=n_features,
            output_horizon_space=horizon_y.shape[1],
        )
        model.summary()

        logger.info("Starting model training")
        history = model.fit(
            x=x,
            y=horizon_y,
            epochs=500,
            batch_size=20,
            validation_split=0.25,
            shuffle=True,
        )

        for epoch in range(len(history.history["loss"])):
            metrics_str = " | ".join(
                f"{metric}={values[epoch]:.6f}"
                for metric, values in history.history.items()
            )
            logger.info(f"Epoch {epoch + 1:02d} | {metrics_str}")
        logger.info("Training completed")

        logger.info("Saving training history plot")
        plot = PlotHistory(history=history).plot_history()
        plot.write_image(str(PLOT_PATH) % str(days_ahead + 1))

        logger.info(f"Saving trained model to {str(MODEL_PATH)%str(days_ahead+1)}")
        model.save(str(MODEL_PATH) % str(days_ahead + 1))
