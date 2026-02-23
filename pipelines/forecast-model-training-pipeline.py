import pickle
from pathlib import Path
from typing import Final

import numpy as np
from sklearn.preprocessing import RobustScaler

from core.utils import PlotHistory
from core.logging import LoggerFactory
from core.model import AutonomusForecastModelArchitecture


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
TRAINING_X_PATH: Final[Path] = BASE_DIR / "dataset" / "training-x.npy"
TRAINING_Y_PATH: Final[Path] = BASE_DIR / "dataset" / "training-y.npy"
ARTIFACT_PATH: Final[Path] = BASE_DIR / "artifacts"

SCALER_PATH: Final[Path] = ARTIFACT_PATH / "feature-scaler.pkl"
MODEL_PATH: Final[Path] = ARTIFACT_PATH / "lme-al-forecast-model-%s-days-ahead.keras"
PLOT_PATH: Final[Path] = ARTIFACT_PATH / "loss-plot-%s-days-ahead.png"


if __name__ == "__main__":
    logger.info("Starting autoencoder training pipeline")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact directory ready: %s", ARTIFACT_PATH)

    logger.info("Loading x data from %s", TRAINING_X_PATH)
    x = np.load(TRAINING_X_PATH)

    logger.info("Loading y data from %s", TRAINING_Y_PATH)
    y = np.load(TRAINING_Y_PATH)

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
    for days_ahead in range(5):
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
            epochs=10,
            batch_size=300,
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
