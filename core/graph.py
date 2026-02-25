import os
import json
import shap
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from core.logging import LoggerFactory
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import Optional, Final, List, Dict
from core.model import AutonomusForecastModelArchitecture
from core.prompts import StrucutredSystemPrompt, StructuredUserPrompt


logger = LoggerFactory().get_logger(__name__)

load_dotenv("./.env")
logger.info("Environment variables loaded")


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

FEATURE_PATH = BASE_DIR / "dataset" / "features.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "lme-al-forecast-model-%s-days-ahead.keras"
SCALER_MODEL_PATH = BASE_DIR / "artifacts" / "feature-scaler.pkl"
FEATURE_MEANING_PATH = BASE_DIR / "artifacts" / "feature-interpretation.json"
BG_DATA_PATH = BASE_DIR / "dataset" / "training-x.npy"


class StructuredInsight(BaseModel):
    feature_name: str
    feature_behavior: str
    shap_interpretation: str
    forecast_alignment: str
    influence_strength: str
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
    shap_score: Optional[Dict] = None
    insight: Optional[StructuredInsight] = None


class Compress:
    def __init__(self):
        pass

    def compress_numeric_payload(obj):
        """
        Convert all numeric values to float32 and round to 2 decimals.
        Recursively supports dict, list, numpy arrays.
        """

        import numpy as np

        if isinstance(obj, dict):
            return {k: Compress.compress_numeric_payload(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [Compress.compress_numeric_payload(v) for v in obj]

        elif isinstance(obj, np.ndarray):
            return np.round(obj.astype(np.float32), 2).tolist()

        elif isinstance(obj, (np.floating, float)):
            return float(np.round(np.float32(obj), 2))

        elif isinstance(obj, (np.integer, int)):
            return int(obj)

        return obj


class MarketAnalyst:
    def __init__(self):
        logger.info("Initializing MarketAnalyst")

        self.llm = self._initialize_llm()
        self.structured_llm = self.llm.with_structured_output(StructuredInsight)

        self.model_artifacts = self._load_models()
        self.scaler_artifact = self._load_scaler()
        self.feature_interpretation = self._load_feature_interpretation()

        self.workflow = StateGraph(StructuredAnalystState)
        self._build_workflow()

        logger.info("MarketAnalyst initialized successfully")

    def _initialize_llm(self):
        model_name = os.getenv("LOCAL_LLM_MODEL")
        logger.info("Loading local LLM: %s", model_name)
        return ChatOllama(model=model_name, temperature=0, format="json")

    def _load_models(self) -> List:
        logger.info("Loading forecast models")

        models = [
            AutonomusForecastModelArchitecture.load(str(MODEL_PATH) % (i + 1))
            for i in range(5)
        ]

        logger.info("Loaded %d forecast models", len(models))
        return models

    def _load_scaler(self):
        logger.info("Loading feature scaler")
        with open(SCALER_MODEL_PATH, "rb") as f:
            scaler = pickle.load(f)

        logger.info("Feature scaler loaded")
        return scaler

    def _load_feature_interpretation(self):
        logger.info("Loading feature interpretation metadata")
        with open(FEATURE_MEANING_PATH, "r") as f:
            data = json.load(f)

        logger.info("Feature interpretation metadata loaded")
        return data

    def _load_data(self, state: StructuredAnalystState):
        logger.info("Loading feature dataset")

        df = pd.read_csv(FEATURE_PATH).sort_values("ssd").reset_index(drop=True)

        self.features = df.drop(columns=["ssd"]).columns.tolist()

        if state.feature_name not in self.features:
            logger.error("Feature not found: %s", state.feature_name)
            raise ValueError(f"Feature '{state.feature_name}' not found")

        df = df[df.ssd <= state.ssd_date].copy()

        if df.empty:
            logger.error("No data available before SSD date: %s", state.ssd_date)
            raise ValueError("No data available before SSD date")

        state.ssd_lme_price = float(df.loc[df.ssd == state.ssd_date, "lme_al"].iloc[0])

        logger.info("SSD LME price: %.2f", state.ssd_lme_price)

        df_last = df.tail(10).reset_index(drop=True)

        x_np = df_last.drop(columns=["ssd"]).to_numpy()
        x_np = self.scaler_artifact.transform(x_np)

        self.x_np = x_np.reshape(1, *x_np.shape)

        logger.info("Model input prepared with shape %s", self.x_np.shape)

        meta = self.feature_interpretation[state.feature_name]

        state.feature_business_name = meta["business_name"]
        state.feature_macro_role = meta["macro_role"]
        state.feature_asset_class = meta["asset_class"]
        state.feature_relation_to_lme_al = meta["relationship_to_aluminum"]

        ts_json = json.loads(
            df_last[["ssd", state.feature_name]].to_json(orient="records")
        )
        state.feature_timeseries = json.dumps(
            Compress.compress_numeric_payload(ts_json)
        )

        return state

    def _get_lme_al_forecasting(self, state: StructuredAnalystState):
        logger.info("Generating LME Aluminum forecasts")

        self.next_dates = (
            pd.date_range(
                start=pd.to_datetime(state.ssd_date) + pd.offsets.BDay(1),
                periods=5,
                freq="B",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )

        forecasts = []

        for model in self.model_artifacts:
            pred = model.predict(self.x_np)[0][0]
            price = round(float(pred * state.ssd_lme_price + state.ssd_lme_price), 2)
            forecasts.append(price)

        forecast_df = pd.DataFrame(
            {
                "future_dates": self.next_dates,
                "forecasted_lme_al_price": forecasts,
            }
        )

        state.lme_al_forecast = json.dumps(
            Compress.compress_numeric_payload(
                json.loads(forecast_df.to_json(orient="records"))
            )
        )

        logger.info("Forecast generation completed")

        return state

    def _get_shap_score(self, state: StructuredAnalystState):
        logger.info("Computing SHAP explanations")

        bg_data = np.load(BG_DATA_PATH)

        idx = np.random.choice(len(bg_data), 100, replace=False)
        bg_data = bg_data[idx]

        n_samples, n_steps, n_features = bg_data.shape

        scaled = self.scaler_artifact.transform(bg_data.reshape(-1, n_features))

        bg_data = scaled.reshape(n_samples, n_steps, n_features)

        f_index = self.features.index(state.feature_name)
        shap_scores = {}

        for date, model in zip(self.next_dates, self.model_artifacts):
            explainer = shap.GradientExplainer(model.model, bg_data)
            shap_val = explainer.shap_values(self.x_np)[0][0][:, f_index]
            shap_scores[f"SHAP score for {date}"] = Compress.compress_numeric_payload(
                shap_val
            )

        state.shap_score = shap_scores

        logger.info("SHAP computation completed")

        return state

    def _get_structural_insights(self, state: StructuredAnalystState):
        logger.info("Generating structured analyst insights")

        response = self.structured_llm.invoke(
            [
                ("system", StrucutredSystemPrompt.prompt),
                (
                    "human",
                    StructuredUserPrompt(
                        ssd_date=state.ssd_date,
                        lme_al_forecast=state.lme_al_forecast,
                        feature_business_name=state.feature_business_name,
                        feature_asset_class=state.feature_asset_class,
                        feature_macro_role=state.feature_macro_role,
                        feature_relation_to_lme_al=state.feature_relation_to_lme_al,
                        feature_timeseries=state.feature_timeseries,
                        shap_score=Compress.compress_numeric_payload(state.shap_score),
                        ssd_lme_price=state.ssd_lme_price,
                    ).get_prompt(),
                ),
            ]
        )
        logger.info(response)
        logger.info(type(response))
        state.insight = response
        logger.info("Insight generation completed")

        return state

    def _build_workflow(self):
        self.workflow.add_node("load_data", self._load_data)
        self.workflow.add_node("forecast", self._get_lme_al_forecasting)
        self.workflow.add_node("shap", self._get_shap_score)
        self.workflow.add_node("insights", self._get_structural_insights)

        self.workflow.set_entry_point("load_data")

        self.workflow.add_edge("load_data", "forecast")
        self.workflow.add_edge("forecast", "shap")
        self.workflow.add_edge("shap", "insights")
        self.workflow.add_edge("insights", END)

        self.app = self.workflow.compile()

        logger.info("LangGraph workflow compiled")

    def get_insights(self, ssd_date: str, feature_name: str):
        logger.info(
            "Executing analyst workflow | date=%s | feature=%s",
            ssd_date,
            feature_name,
        )

        result = self.app.invoke({"ssd_date": ssd_date, "feature_name": feature_name})

        logger.info("Workflow execution finished")

        return result
