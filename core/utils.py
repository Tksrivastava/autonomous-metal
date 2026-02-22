import os
import sqlite3
import tensorflow as tf
from pathlib import Path
from typing import Final
import plotly.graph_objects as go

import pandas as pd
from dotenv import load_dotenv

from core.logging import LoggerFactory


logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


FILE_PATH: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = f"{FILE_PATH}/.env"
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
    - Intended for visualization only; it does not return numerical results."""

    def __init__(self, history: tf.keras.callbacks.History = None):
        self.history = history

    def plot_history(self):
        hist = self.history.history
        epochs = list(range(1, len(hist["loss"]) + 1))

        fig = go.Figure()

        # Training loss
        fig.add_trace(
            go.Scatter(
                x=epochs, y=hist["loss"], mode="lines+markers", name="Train Loss"
            )
        )

        # Validation loss (if exists)
        if "val_loss" in hist:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_loss"],
                    mode="lines+markers",
                    name="Validation Loss",
                )
            )

        fig.update_layout(
            title="History",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            hovermode="x unified",
        )
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
        df = df.groupby("ssd", as_index=False).agg({"lme_al": "mean"})
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

            temp = pd.DataFrame(
                {
                    "ssd": df["ssd"],
                    "current_spot_price": df["lme_al"],
                    "predicted_for": future_date,
                    "days_ahead": days,
                    "y": (future_price - df["lme_al"]) / df["lme_al"],
                }
            )

            records.append(temp)

        label_df = pd.concat(records, ignore_index=True)

        logger.info("Labels created: %s rows", len(label_df))
        return label_df


class FetchRawFeatures:
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
        df = df.groupby("ssd", as_index=False).agg({"lme_al": "mean"})
        logger.info("Spot prices loaded: %s rows", len(df))
        return df

    def _load_gfinance_index(self) -> pd.DataFrame:
        """Load and prepare Google Finance data."""
        query = """
            SELECT
                date AS ssd,
                "index",
                CAST(index_value AS REAL) AS index_value
            FROM "google-finance-index"
            WHERE "index"!="alcoa_corp_stocks"
            ORDER BY date
        """
        df = pd.read_sql(query, self.conn, parse_dates=["ssd"])
        logger.info("Google Finance data loaded: %s rows", len(df))
        df = (
            df.pivot_table(
                index="ssd", columns="index", values="index_value", aggfunc="mean"
            )
            .reset_index()
            .sort_values("ssd")
            .reset_index(drop=True)
        )
        logger.info("Google Finance data pivoted: %s rows", len(df))
        df = df.ffill().bfill()
        logger.info("Few data points filled")
        return df

    def _load_inventory_level(self) -> pd.DataFrame:
        """Load and prepare LME Aluminum inventory data"""
        query = """
            SELECT
                date AS ssd,
                CAST(lme_al_inventory AS REAL) AS lme_al_inventory
            FROM "lme-aluminum-daily-inventory"
            ORDER BY date
        """
        df = pd.read_sql(query, self.conn, parse_dates=["ssd"])
        logger.info("LME Aluminum Inventory data loaded: %s rows", len(df))
        return df

    def _load_baltic_index(self) -> pd.DataFrame:
        """Load and prepare LME Aluminum inventory data"""
        query = """
            SELECT
                date AS ssd,
                CAST(baltic_dry_index AS REAL) AS baltic_dry_index
            FROM "baltic-dry-index"
            ORDER BY date
        """
        df = pd.read_sql(query, self.conn, parse_dates=["ssd"])
        logger.info("Baltic Dry index data loaded: %s rows", len(df))
        return df

    def fetch(self) -> pd.DataFrame:
        spot = self._load_spot_prices()
        gf = self._load_gfinance_index()
        inv = self._load_inventory_level()
        bal = self._load_baltic_index()

        df = (
            spot.merge(gf, on="ssd", how="outer")
            .merge(inv, on="ssd", how="outer")
            .merge(bal, on="ssd", how="outer")
            .ffill()
            .bfill()
        )
        logger.info(f"All raw features merged - {df.shape}")
        return df


class RAGQuestionnaire:
    question_set = {
        "Raw materials / Input costs": [
            "Are there changes in electricity prices affecting aluminium smelters?",
            "Have power shortages or energy rationing been reported?",
            "Are energy subsidies or tariffs affecting metal producers?",
            "Are oil, gas, or coal price movements impacting production economics?",
            "Are alumina prices rising or falling?",
            "Are bauxite mining operations disrupted or expanded?",
            "Are refinery outages affecting alumina supply?",
            "Have smelters announced cost pressures due to energy inputs?",
            "Are carbon pricing or emissions costs affecting aluminium production?",
            "Are renewable energy transitions impacting smelting costs?",
            "Are producers shutting capacity due to high operating costs?",
            "Are input material transportation costs changing?",
        ],
        "Geo-political": [
            "Are there conflicts affecting aluminium-producing regions?",
            "Are sanctions imposed on aluminium producers or exporting countries?",
            "Are geopolitical tensions affecting commodity markets broadly?",
            "Are trade routes disrupted due to political instability?",
            "Are shipping lanes or ports affected by conflict?",
            "Are diplomatic disputes influencing metal exports?",
            "Are strategic metals being restricted for national security reasons?",
            "Are geopolitical risks affecting investor sentiment toward metals?",
        ],
        "Government policy": [
            "Have governments introduced tariffs on aluminium imports or exports?",
            "Are there anti-dumping investigations involving aluminium?",
            "Are new environmental regulations affecting smelters?",
            "Are subsidies announced for domestic aluminium production?",
            "Are industrial policies promoting local metal manufacturing?",
            "Are carbon border taxes or CBAM-like policies mentioned?",
            "Are mining licenses approved or revoked?",
            "Are export taxes or quotas introduced?",
            "Are governments supporting infrastructure or construction stimulus?",
            "Are regulatory approvals delaying mining or refining projects?",
        ],
        "Supply chain": [
            "Are logistics disruptions affecting metal shipments?",
            "Are ports congested or experiencing delays?",
            "Are freight or shipping costs changing significantly?",
            "Are rail or trucking bottlenecks reported?",
            "Are refinery or smelter outages affecting deliveries?",
            "Are warehouse or storage constraints mentioned?",
            "Are supply disruptions caused by labor strikes?",
            "Are weather events disrupting transportation?",
            "Are companies reporting delivery delays to buyers?",
        ],
        "Inventory": [
            "Are LME aluminium inventories rising or falling?",
            "Are warehouse stock cancellations increasing?",
            "Are off-exchange inventories discussed?",
            "Are stock drawdowns reported?",
            "Are inventories shifting between regions?",
            "Are traders moving metal between warehouses?",
            "Are stockpiles building due to weak demand?",
            "Are shortages reported in physical markets?",
            "Are financing deals affecting inventory storage?",
            "Are inventory premiums changing?",
        ],
        "Trade flow": [
            "Are aluminium exports increasing from major producers?",
            "Are imports rising in key consuming countries?",
            "Are trade flows shifting between regions?",
            "Are sanctions redirecting aluminium shipments?",
            "Are arbitrage opportunities changing trade routes?",
            "Is Chinese aluminium export activity changing?",
            "Are regional price spreads influencing flows?",
            "Are new trade agreements affecting metals trade?",
            "Are tariffs changing sourcing behavior?",
            "Are supply being redirected due to demand weakness elsewhere?",
        ],
        "Physical demand": [
            "Is demand from construction sector rising or falling?",
            "Are automotive manufacturers changing aluminium usage?",
            "Are EV production trends affecting aluminium demand?",
            "Are aerospace orders increasing or declining?",
            "Are packaging or beverage can demand trends mentioned?",
            "Are infrastructure projects increasing metal consumption?",
            "Are manufacturing PMIs indicating stronger or weaker demand?",
            "Are downstream fabricators reducing orders?",
            "Are buyers delaying purchases?",
            "Are seasonal demand patterns discussed?",
            "Are demand slowdowns reported in China, Europe, or US?",
            "Are premiums indicating tight or weak physical demand?",
        ],
        "Technology": [
            "Are new smelting technologies introduced?",
            "Are low-carbon aluminium initiatives discussed?",
            "Are recycling technologies expanding supply?",
            "Are efficiency improvements reducing production costs?",
            "Are alternative materials replacing aluminium?",
            "Are battery or EV technologies increasing aluminium intensity?",
            "Are green aluminium certifications affecting market demand?",
        ],
        "Macro / Financial factors": [
            "Is US dollar strength or weakness affecting commodities?",
            "Are interest rate changes impacting metals markets?",
            "Are recession fears discussed?",
            "Are inflation expectations influencing commodities?",
            "Are commodity funds increasing or reducing exposure?",
            "Are investors rotating away from industrial metals?",
            "Are global growth forecasts revised?",
            "Are central bank policies impacting risk sentiment?",
            "Are equity or commodity indices moving broadly?",
            "Are hedge funds changing positioning?",
            "Are currency fluctuations affecting producer competitiveness?",
            "Are risk-off or risk-on market environments mentioned?",
        ],
    }
