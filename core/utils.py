import os
import sqlite3
import tensorflow as tf
from pathlib import Path
import plotly.graph_objects as go

import pandas as pd
from dotenv import load_dotenv

from core.logging import LoggerFactory


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


FILE_PATH = Path(__file__).resolve().parents[1]
ENV_PATH = f"{FILE_PATH}/.env"
load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env")

class FetchFromKaggle:
    """
    Handles authentication and dataset download from Kaggle.

    Responsibilities:
    - Authenticate with Kaggle using environment credentials
    - Download and unzip datasets
    - Ensure consistent local storage path

    Assumes:
    - KAGGLE_USERNAME and KAGGLE_KEY are present in .env
    - KAGGLE_DATASET is defined in .env
    """

    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.logger = logger
        self.api = KaggleApi()

        self.logger.info("Authenticating with Kaggle API")
        self.api.authenticate()

        self.save_path = FILE_PATH / "dataset"
        self._create_download_path()

        self.logger.info(
            "Kaggle connection established. Download path: %s", self.save_path
        )

    def _create_download_path(self) -> None:
        """
        Creates the dataset download directory if it does not exist.
        """
        os.makedirs(self.save_path, exist_ok=True)
        self.logger.debug("Dataset directory ensured at %s", self.save_path)

    def download(self) -> None:
        """
        Downloads and extracts the Kaggle dataset specified in the environment.
        """
        dataset_name = os.getenv("KAGGLE_DATASET")
        if not dataset_name:
            raise ValueError("KAGGLE_DATASET not found in environment variables")

        self.logger.info("Starting dataset download: %s", dataset_name)

        self.api.dataset_download_files(
            dataset=dataset_name,
            path=self.save_path,
            unzip=True,
            force=True,
        )

        self.logger.info("Dataset downloaded and extracted successfully")
    
class PlotHistory(object):
    """
Utility class for visualizing training history of TensorFlow / Keras models.

This class provides a lightweight wrapper around a `tf.keras.callbacks.History`
object and generates interactive loss curves using Plotly. It is intended for
quick inspection of training dynamics such as convergence behavior and
overfitting.

Currently supported metrics:
- Training loss (`loss`)
- Validation loss (`val_loss`), if available

The visualization is rendered as an interactive Plotly figure with epoch-wise
traces, suitable for exploratory analysis in notebooks or local development
environments.

Parameters
----------
history : tf.keras.callbacks.History, optional
    History object returned by `model.fit`. Must contain a `history` attribute
    with recorded loss values.

Notes
-----
- This utility assumes that the model was compiled with a loss function.
- Validation loss is plotted only if `val_loss` is present in the history.
- The class does not perform input validation and will raise an error if an
    invalid or incomplete History object is provided.
- Intended for visualization only; it does not return numerical results.
"""
    def __init__(self, history: tf.keras.callbacks.History = None):
        self.history = history
    def plot_history(self):
        hist = self.history.history
        epochs = list(range(1, len(hist['loss']) + 1))

        fig = go.Figure()

        # Training loss
        fig.add_trace(go.Scatter(x=epochs, y=hist['loss'], mode='lines+markers', name='Train Loss'))

        # Validation loss (if exists)
        if 'val_loss' in hist:
            fig.add_trace(go.Scatter(x=epochs, y=hist['val_loss'], mode='lines+markers', name='Validation Loss'))

        fig.update_layout(title='History', xaxis_title='Epoch',
                        yaxis_title='Loss', template='plotly_white', hovermode='x unified')
        fig.show()
        return fig

class PrepareLabels:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    def _load_spot_prices(self) -> pd.DataFrame:
        """Load and prepare LME Aluminum spot prices."""
        query = """
            SELECT
                date AS ssd,
                CAST(lme_al AS REAL) AS lme_al
            FROM "lme-aluminum-spot-prices"
            ORDER BY date
        """
        df = pd.read_sql(query, self.conn, parse_dates=["ssd"])
        logger.info("Spot prices loaded: %s rows", len(df))
        return df
    def build_labels(self, horizon: int) -> pd.DataFrame:
        """
        Create future return labels for given horizon.
        Output columns:
            ssd, current_spot_price, predicted_for, days_ahead, y
        """
        df = self._load_spot_prices()
        logger.info(f"LME Aluminum spot prices data fetched - {df.shape}")
        df.sort_values("ssd", inplace=True)
        df.reset_index(drop=True, inplace=True)

        records = []

        for days in range(1, horizon + 1):
            future_price = df["lme_al"].shift(-days)
            future_date = df["ssd"].shift(-days)

            temp = pd.DataFrame({
                "ssd": df["ssd"],
                "current_spot_price": df["lme_al"],
                "predicted_for": future_date,
                "days_ahead": days,
                "y": (future_price - df["lme_al"]) / df["lme_al"],
            })

            records.append(temp)

        label_df = pd.concat(records, ignore_index=True)

        logger.info("Labels created: %s rows", len(label_df))
        return label_df