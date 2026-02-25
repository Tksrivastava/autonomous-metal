import os
import pickle
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Final, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# from core.prompts import LLMPrompts
from core.logging import LoggerFactory
from core.model import AutonomusForecastModelArchitecture
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

# Initializing logger
logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)

load_dotenv(dotenv_path="./.env")
logger.info("Environment variables loaded from .env")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
FEATURE_PATH: Final[Path] = BASE_DIR / "dataset" / "features.csv"
MODEL_PATH: Final[Path] = (
    BASE_DIR / "artifacts" / "lme-al-forecast-model-%s-days-ahead.keras"
)
SCALER_MODEL_PATH: Final[Path] = BASE_DIR / "artifacts" / "feature-scaler.pkl"
FEATURE_MEANING_PATH: Final[Path] = (
    BASE_DIR / "artifacts" / "feature-interpretation.json"
)


class StructuredInsight(BaseModel):
    feature_name: str
    feature_behavior: str
    forecast_alignment: str
    analyst_explanation: str


class StructuredAnalystState(BaseModel):
    ssd_date: str
    feature_name: str
    ssd_lme_price: Optional[float] = None
    feature_business_name: Optional[str] = None
    feature_asset_class: Optional[str] = None
    feature_macro_role: Optional[str] = None
    feature_relation_to_lme_al: Optional[str] = None
    lme_al_forecast: Optional[str] = None
    feature_timeseries: Optional[str] = None
    shap_score: Optional[str] = None
    insight: Optional[StructuredInsight] = None


class MarketAnalyst:
    def __init__(self, local_inference: bool = True):
        self.local_inference = local_inference
        self.llm = self._initialize_llm()
        logger.info("LLM loaded successfully")
        self.workflow = StateGraph(StructuredAnalystState)
        self._build_workflow()
        logger.info("LangGraph workflow created")
        self.model_artifact = self._load_models()
        self.scaler_artifact = self._load_scaler()
        self.feature_interpretation = self._load_feature_interpretation()

    def _initialize_llm(self):
        if self.local_inference:
            model_name = os.getenv("LOCAL_LLM_MODEL")
            logger.info(f"Initializing local LLM: {model_name}")
            return ChatOllama(model=model_name, temperature=0, format="json")
        model_name = os.getenv("GROQ_LLM_MODEL")
        logger.info(f"Initializing Groq LLM: {model_name}")
        return ChatGroq(
            model=model_name, api_key=os.getenv("GROQ_API_KEY"), temperature=0
        )

    def _load_models(self) -> List:
        logger.info("Loading forecast model artifacts")
        models = []
        for index in range(5):
            path = str(MODEL_PATH) % str(index + 1)
            models.append(AutonomusForecastModelArchitecture.load(path))
        logger.info(f"{len(models)} models loaded")
        return models

    def _load_scaler(self):
        logger.info("Loading scaler artifact")
        with open(SCALER_MODEL_PATH, "rb") as f:
            scaler = pickle.load(f)
        return scaler

    def _load_feature_interpretation(self):
        logger.info("Loading feature interpretation json")
        with open(FEATURE_MEANING_PATH, "r") as f:
            return json.load(f)

    def _load_data(self, state: StructuredAnalystState) -> StructuredAnalystState:
        logger.info("Loading feature dataset")
        df = (
            pd.read_csv(FEATURE_PATH)
            .sort_values("ssd", ascending=True)
            .reset_index(drop=True)
        )
        logger.info(f"Feature data loaded — shape={df.shape}")
        # Filter until SSD date
        df = df.loc[df.ssd <= state.ssd_date].copy()
        logger.info("Dataset filtered by SSD date")
        # Current date price
        state.ssd_lme_price = float(
            df.loc[df.ssd == state.ssd_date]["lme_al"].values[0]
        ) / round(2)
        logger.info("SSD price added in State")
        # Last 10 observations
        df = df.iloc[-10:].reset_index(drop=True)
        # Convert to numpy
        self.x_np = df.drop(columns=["ssd"]).to_numpy()
        self.x_np = self.scaler_artifact.transform(self.x_np).reshape(
            1, self.x_np.shape[0], self.x_np.shape[1]
        )
        logger.info(f"Prepared model input shape={self.x_np.shape}")
        # Filter features information
        state.feature_timeseries = df[["ssd", state.feature_name]].to_toon()
        state.feature_business_name = self.feature_interpretation[state.feature_name][
            "business_name"
        ]
        state.feature_macro_role = self.feature_interpretation[state.feature_name][
            "macro_role"
        ]
        state.feature_asset_class = self.feature_interpretation[state.feature_name][
            "asset_class"
        ]
        state.feature_relation_to_lme_al = self.feature_interpretation[
            state.feature_name
        ]["relationship_to_aluminum"]
        logger.info("Feature related State updated")
        return state

    def _get_lme_al_forecasting(self, state: StructuredAnalystState):
        # Future dates
        next_dates = (
            pd.date_range(
                start=pd.to_datetime(state.ssd_date) + pd.offsets.BDay(1),
                periods=5,
                freq="B",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )
        logger.info("Future dates generated")
        state.lme_al_forecast = pd.DataFrame(
            {
                "future_dates": next_dates,
                "forecasted_lme_al_price": [
                    round(
                        float(
                            model.predict(self.x_np)[0][0] * state.ssd_lme_price
                            + state.ssd_lme_price
                        ),
                        2,
                    )
                    for model in self.model_artifact
                ],
            }
        ).to_toon()
        logger.info("LME Aluminum Forecasting generated")
        return state

    def _build_workflow(self):
        self.workflow.add_node("Load Informations", self._load_data)
        self.workflow.add_node(
            "Generate LME Aluminum predictions", self._get_lme_al_forecasting
        )

        self.workflow.set_entry_point("Load Informations")

        self.workflow.add_edge("Load Informations", "Generate LME Aluminum predictions")
        self.workflow.add_edge("Generate LME Aluminum predictions", END)

        self.app = self.workflow.compile()
        logger.info("Workflow compiled successfully")

    def get_report(self, ssd_date: str, feature_name: str):
        logger.info("Graph execution started")
        result = self.app.invoke({"ssd_date": ssd_date, "feature_name": feature_name})
        logger.info("Graph execution completed")
        return result
