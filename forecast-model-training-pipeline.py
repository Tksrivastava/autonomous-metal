import os
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


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
TRAINIGN_DATA_PATH: Final[Path] = BASE_DIR / "dataset" / "training-data.npz"
ARTIFACT_PATH: Final[Path] = BASE_DIR / "artifacts"

SCALER_PATH: Final[Path] = ARTIFACT_PATH / "feature-scaler.pkl"
MODEL_PATH: Final[Path] = ARTIFACT_PATH / "autoencoder-model.keras"
PLOT_PATH: Final[Path] = ARTIFACT_PATH / "loss-plot.png"

os.makedirs(ARTIFACT_PATH, exist_ok=True)
logger.debug("Artifacts directory ensured")


def main() -> None:
    logger.info("Starting autoencoder training pipeline")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Artifact directory ready: %s", ARTIFACT_PATH)

    logger.info("Loading features from %s", TRAINIGN_DATA_PATH)
    data = np.load(TRAINIGN_DATA_PATH)

    x, y = data["X"], data["y"]
    logger.info(f"Seperating x, y - {x.shape}, {y.shape}")

    logger.info("Applying RobustScaler")
    scaler = RobustScaler()

    n_samples, n_steps, n_features = x.shape
    x = x.reshape(-1, n_features) ## Converting x to 2D
    x = scaler.fit_transform(x).reshape(n_samples, n_steps, n_features) ## Converting to 3D after scaling

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Feature scaler saved to %s", SCALER_PATH)

    logger.info(
        "Initializing autoencoder | input_space=%d | latent_space=%d | seed=%d",
        x.shape[1], 10, 42
    )
    
    model = AutonomusForecastModelArchitecture(
        seed=42,
        input_horizon_space=n_steps,
        input_feature_space=n_features,
        output_horizon_space=y.shape[1]
    )

    model.summary()

    logger.info("Starting model training")

    history = model.fit(
        x=x,
        y=y,
        epochs=30,
        batch_size=300,
        validation_split=0.35,
        shuffle=False
    )
    
    for epoch in range(len(history.history["loss"])):
        metrics_str = " | ".join(f"{metric}={values[epoch]:.6f}"
                                 for metric, values in history.history.items())
        logger.info(f"Epoch {epoch + 1:02d} | {metrics_str}")


    logger.info("Training completed")

    logger.info("Saving training history plot")
    plot = PlotHistory(history=history).plot_history()
    plot.write_image(PLOT_PATH)

    logger.info(f"Saving trained model to {MODEL_PATH}")
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()