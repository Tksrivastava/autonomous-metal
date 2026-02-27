import glob
import json
import shap
import time
import pickle
import numpy as np
import pandas_toon
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from typing import Optional, List
from langchain_groq import ChatGroq
from core.logging import LoggerFactory
from langgraph.graph import StateGraph, END
from core.model import AutonomousForecastModelArchitecture
from langchain_core.output_parsers import PydanticOutputParser
from core.prompts import (
    StructuredSystemPrompt,
    StructuredUserPrompt,
    StructuredAnalystReportPrompt,
)

logger = LoggerFactory().get_logger(__name__)


class FeatureInsight(BaseModel):
    feature_name: str
    feature_behavior: str
    shap_interpretation: str
    forecast_alignment: str
    influence_strength: str
    analyst_explanation: str


class FeatureAnalystState(BaseModel):
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
    insight: Optional[FeatureInsight] = None


class StructuredFeatureMarketAnalyst:
    def __init__(
        self,
        groq_model: str = None,
        groq_api: str = None,
        scaler_artifact_path: str = None,
        forecast_models_artifact_directory_path: str = None,
        feature_order_path: str = None,
        feature_interpretation_path: str = None,
        spot_price_path: str = None,
        ssd_date: str = None,
        x_data_path: str = None,
        raw_features_path: str = None,
        feature_list: List[str] = None,
        horizon: int = 5,
    ):

        self.groq_model = groq_model
        self.groq_api = groq_api
        self.scaler_artifact_path = scaler_artifact_path
        self.forecast_models_artifact_directory_path = (
            forecast_models_artifact_directory_path
        )
        self.feature_order_path = feature_order_path
        self.feature_interpretation_path = feature_interpretation_path
        self.spot_price_path = spot_price_path
        self.x_data_path = x_data_path
        self.raw_features_path = raw_features_path

        self.ssd_date = (
            ssd_date if ssd_date is not None else self._raise("Please provide ssd_date")
        )

        self.feature_list = (
            feature_list
            if feature_list is not None
            else self._raise("Please provide feature_list")
        )

        self.workflow = StateGraph(FeatureAnalystState)
        self.horizon = horizon

        self._init_llm()
        self._init_scaler()
        self._init_forecast_models()
        self._init_feature_order()
        self._init_feature_interpretation()
        self._init_spot_price()
        self._init_x_data()
        self._init_shap_explainers()
        self._init_raw_features()
        self._init_future_dates()
        self._init_current_spot_price()
        self._init_build_workflow()

        self.parser = PydanticOutputParser(pydantic_object=FeatureInsight)

    def _raise(self, comment: str):
        raise ValueError(comment)

    def _init_llm(self):
        if not self.groq_model or not self.groq_api:
            raise ValueError("Please provide groq_model and groq_api")

        self.llm = ChatGroq(
            model=self.groq_model,
            api_key=self.groq_api,
            temperature=0,
        )
        logger.info(f"LLM loaded | model={self.groq_model}")

    def _init_scaler(self):
        if self.scaler_artifact_path is None:
            raise ValueError("Please provide scaler_artifact_path")
        self._load_scaler_artifact()

    def _init_forecast_models(self):
        if self.forecast_models_artifact_directory_path is None:
            raise ValueError("Please provide forecast_models_artifact_directory_path")
        self._load_forecast_model_artifacts()

    def _init_feature_order(self):
        if self.feature_order_path is None:
            raise ValueError("Please provide feature_order_path")

        with open(self.feature_order_path, "rb") as f:
            self.feature_order = pickle.load(f)

        logger.info("Feature order loaded")

    def _init_feature_interpretation(self):
        if self.feature_interpretation_path is None:
            raise ValueError("Please provide feature_interpretation_path")

        with open(self.feature_interpretation_path, "r") as f:
            self.feature_interpretation = json.load(f)

        logger.info("Feature interpretation loaded")

    def _init_spot_price(self):
        if self.spot_price_path is None:
            raise ValueError("Please provide spot_price_path")

        self.spot_price = pd.read_csv(self.spot_price_path)
        logger.info("Spot price data loaded")

    def _init_x_data(self):
        if self.x_data_path is None:
            raise ValueError("Please provide x_data_path")

        with open(self.x_data_path, "rb") as f:
            self.x = pickle.load(f)

        logger.info("Model input (X) data loaded")
        self._load_background_data_shap()

    def _init_shap_explainers(self):
        self.shap_explainers = [
            shap.GradientExplainer(model.model, self.bg_data) for model in self.models
        ]
        logger.info("SHAP explainers initialized")

    def _init_raw_features(self):
        if self.raw_features_path is None:
            raise ValueError("Please provide raw_features_path")

        self.raw_features = pd.read_csv(self.raw_features_path)
        logger.info("Raw feature dataset loaded")

    def _init_future_dates(self):
        self.next_dates = (
            pd.date_range(
                start=pd.to_datetime(self.ssd_date) + pd.offsets.BDay(1),
                periods=self.horizon,
                freq="B",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )
        logger.info("Future business dates generated")

    def _init_current_spot_price(self):

        filtered = self.spot_price.loc[self.spot_price.ssd == self.ssd_date]

        if filtered.empty:
            raise ValueError(f"No spot price found for date {self.ssd_date}")

        self.current_spot = round(filtered["current_spot_price"].values[0], 2)

        logger.info(f"Current spot price loaded | price={self.current_spot}")

    def _load_scaler_artifact(self):
        with open(self.scaler_artifact_path, "rb") as f:
            self.scaler_artifact = pickle.load(f)
        logger.info("Scaler artifact loaded")

    def _load_forecast_model_artifacts(self):

        artifacts = sorted(
            glob.glob(f"{self.forecast_models_artifact_directory_path}/*.keras")
        )

        if not artifacts:
            raise ValueError("No .keras files found in forecast directory")

        logger.info(f"{len(artifacts)} forecast model artifacts found")

        self.models = [
            AutonomousForecastModelArchitecture.load(path=path) for path in artifacts
        ]

        logger.info("All forecast models loaded successfully")

    def _load_background_data_shap(self):

        self.bg_data = np.stack(list(self.x.values())).astype(np.float32)

        sample_size = min(100, len(self.bg_data))
        idx = np.random.choice(len(self.bg_data), sample_size, replace=False)
        self.bg_data = self.bg_data[idx]

        logger.info("Background SHAP data sampled")

        n_samples, n_steps, n_features = self.bg_data.shape

        self.bg_data = self.scaler_artifact.transform(
            self.bg_data.reshape(-1, n_features)
        ).reshape(n_samples, n_steps, n_features)

        logger.info("Background SHAP data scaled")

    def _workflow_get_feature_information(self, state: FeatureAnalystState):
        logger.info("Node started: get_feature_information")

        meta_data = self.feature_interpretation[state.feature_name]

        state.feature_business_name = meta_data["business_name"]
        state.feature_macro_role = meta_data["macro_role"]
        state.feature_asset_class = meta_data["asset_class"]
        state.feature_relation_to_lme_al = meta_data["relationship_to_aluminum"]

        logger.info("Node completed: get_feature_information")
        return state

    def _workflow_get_forecasting(self, state: FeatureAnalystState):
        logger.info("Node started: get_forecasting")

        self.sub_x = self.x[state.ssd_date]
        self.sub_x = self.scaler_artifact.transform(self.sub_x).reshape(
            1, *self.sub_x.shape
        )

        forecasts = [round(model.predict(self.sub_x)[0][0], 2) for model in self.models]

        logger.info(f"Return forecasts generated | values={forecasts}")

        forecasts = [
            round((self.current_spot * r) + self.current_spot) for r in forecasts
        ]

        self.forecasts = pd.DataFrame(
            {"FutureDates": self.next_dates, "ForecastedPrice": forecasts}
        )

        state.ssd_lme_price = self.current_spot
        state.lme_al_forecast = self.forecasts.to_toon()

        logger.info("Node completed: get_forecasting")
        return state

    def _workflow_get_feature_timeseries(self, state: FeatureAnalystState):
        logger.info("Node started: get_feature_timeseries")

        self.feature_timeseries = (
            self.raw_features.loc[self.raw_features.ssd <= self.ssd_date]
            .sort_values("ssd")
            .reset_index(drop=True)
            .iloc[-self.bg_data.shape[1] :][["ssd", self.feature_name]]
            .reset_index(drop=True)
            .round({self.feature_name: 2})
        )

        state.feature_timeseries = self.feature_timeseries.to_toon()

        logger.info("Node completed: get_feature_timeseries")
        return state

    def _workflow_get_shap_scores(self, state: FeatureAnalystState):
        logger.info("Node started: get_shap_scores")

        feature_index = self.feature_order.index(state.feature_name)

        shap_scores = [
            exp.shap_values(self.sub_x)[0][0][:, feature_index]
            for exp in self.shap_explainers
        ]

        columns = [f"SHAPScoreFor {d}" for d in self.next_dates]

        shap_scores = pd.DataFrame(zip(columns, shap_scores))
        shap_scores["ssd"] = self.feature_timeseries["ssd"]
        shap_scores = shap_scores.round(4)
        shap_scores.columns = [str(c) for c in shap_scores.columns]

        state.shap_score = shap_scores.to_toon()

        logger.info("Node completed: get_shap_scores")
        return state

    def _workflow_get_llm_insight(self, state: FeatureAnalystState):
        logger.info("Node started: get_llm_insight")

        user_prompt = StructuredUserPrompt(
            ssd_date=state.ssd_date,
            lme_al_forecast=state.lme_al_forecast or "",
            feature_business_name=state.feature_business_name or "",
            feature_asset_class=state.feature_asset_class or "",
            feature_macro_role=state.feature_macro_role or "",
            feature_relation_to_lme_al=state.feature_relation_to_lme_al or "",
            feature_timeseries=state.feature_timeseries or "",
            shap_score=state.shap_score or "",
            ssd_lme_price=state.ssd_lme_price or 0.0,
        ).get_prompt()

        messages = [
            ("system", StructuredSystemPrompt.prompt),
            ("user", user_prompt + "\n\n" + self.parser.get_format_instructions()),
        ]

        response = self.llm.invoke(messages)

        try:
            response_dict = json.loads(response.content)
            state.insight = FeatureInsight(**response_dict)
        except Exception:
            state.insight = self.parser.parse(response.content)

        logger.info("Node completed: get_llm_insight")
        return state

    def _init_build_workflow(self):

        logger.info("Building LangGraph workflow")

        self.workflow.add_node(
            "get_feature_information", self._workflow_get_feature_information
        )
        self.workflow.add_node("get_forecasting", self._workflow_get_forecasting)
        self.workflow.add_node(
            "get_feature_timeseries", self._workflow_get_feature_timeseries
        )
        self.workflow.add_node("get_shap_scores", self._workflow_get_shap_scores)
        self.workflow.add_node("get_llm_insight", self._workflow_get_llm_insight)
        logger.info("Workflow nodes registered")

        self.workflow.set_entry_point("get_feature_information")
        logger.info("Workflow entry point set")

        self.workflow.add_edge("get_feature_information", "get_feature_timeseries")
        self.workflow.add_edge("get_feature_timeseries", "get_forecasting")
        self.workflow.add_edge("get_forecasting", "get_shap_scores")
        self.workflow.add_edge("get_shap_scores", "get_llm_insight")
        self.workflow.add_edge("get_llm_insight", END)
        logger.info("Workflow edges connected")

        self.app = self.workflow.compile()
        logger.info("LangGraph workflow compiled successfully")

    def _get_insights(self):

        responses = []

        for feature_name in tqdm(
            self.feature_list,
            desc="Generating insights",
            unit="feature",
        ):
            self.feature_name = feature_name

            logger.info(
                f"Workflow execution started | date={self.ssd_date} | feature={feature_name}"
            )

            response = self.app.invoke(
                {"ssd_date": self.ssd_date, "feature_name": feature_name}
            )

            responses.append(response["insight"].model_dump())
            logger.info("Insight appended to master response list")

            time.sleep(30)

        return responses

    def get_analyst_report_on_structured_inputs(
        self, forecast_horizon: str = "Next 5 Business Days"
    ):

        response = self._get_insights()
        logger.info("All feature insights generated")

        prompt = StructuredAnalystReportPrompt(
            forecast_horizon=forecast_horizon,
            ssd_date=self.ssd_date,
            current_price=self.current_spot,
            forecast_prices=self.forecasts.to_toon(),
            feature_context=response,
        ).get_prompt()

        time.sleep(30)

        response = self.llm.invoke(prompt)
        logger.info("Final analyst report generated")

        return response.content
