import glob
import json
import shap
import pickle
import numpy as np
import pandas_toon
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from core.logging import LoggerFactory
from langgraph.graph import StateGraph, END
from core.model import AutonomusForecastModelArchitecture
from langchain_core.output_parsers import PydanticOutputParser
from core.prompts import StructuredSystemPrompt, StructuredUserPrompt

logger = LoggerFactory().get_logger(__name__)


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
    shap_score: Optional[str] = None
    insight: Optional[StructuredInsight] = None


class StructuredMarketAnalyst:
    def __init__(
        self,
        gorq_model: str = None,
        gorq_api: str = None,
        scaler_artifact_path: str = None,
        forecast_models_artifact_directory_path: str = None,
        feature_order_path: str = None,
        feature_interpretation_path: str = None,
        spot_price_path: str = None,
        ssd_date: str = None,
        x_data_path: str = None,
        raw_features_path: str = None,
        feature_name: str = None,
        horizon: int = 5
    ):
        # Store parameters
        self.gorq_model = gorq_model
        self.gorq_api = gorq_api
        self.scaler_artifact_path = scaler_artifact_path
        self.forecast_models_artifact_directory_path = (
            forecast_models_artifact_directory_path
        )
        self.feature_order_path = feature_order_path
        self.feature_interpretation_path = feature_interpretation_path
        self.spot_price_path = spot_price_path
        self.ssd_date = (
            ssd_date
            if ssd_date is not None
            else self._raise(comment="Please provide ssd_date")
        )
        self.x_data_path = x_data_path
        self.raw_features_path = raw_features_path
        self.feature_name = (
            feature_name
            if ssd_date is not None
            else self._raise(comment="Please provide feature_name")
        )
        self.workflow = StateGraph(StructuredAnalystState)

        # Initialize required components
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
        self.parser = PydanticOutputParser(pydantic_object=StructuredInsight)

    def _raise(self, comment: str):
        raise ValueError(comment)

    def _init_llm(self):
        if self.gorq_model is None or self.gorq_api is None:
            raise ValueError("Please provide gorq_model and gorq_api")
        self.llm = ChatGroq(model=self.gorq_model, api_key=self.gorq_api, temperature=0)
        logger.info(f"{self.gorq_model} loaded")

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
        logger.info("Spot prices loaded")

    def _init_x_data(self):
        if self.x_data_path is None:
            raise ValueError("Please provide x_data_path")
        with open(self.x_data_path, "rb") as f:
            self.x = pickle.load(f)
        logger.info("X data loaded")
        self._load_background_data_shap()

    def _init_shap_explainers(self):
        self.shap_explainers = [
            shap.GradientExplainer(model.model, self.bg_data) for model in self.models
        ]
        logger.info("SHAP explainers created")

    def _init_raw_features(self):
        if self.raw_features_path is None:
            raise ValueError("Please provide raw_features_path")
        self.raw_features = pd.read_csv(self.raw_features_path)
        logger.info("Raw features loaded")

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
        logger.info("Future dates estimated")

    def _init_current_spot_price(self):
        self.current_spot = round(
            self.spot_price.loc[self.spot_price.ssd == self.ssd_date][
                "current_spot_price"
            ].values[0],
            2,
        )
        logger.info("Current spot prices loaded")

    def _init_build_workflow(self):
        self.workflow.add_node(
            "get_feature_information", self._workflow_get_feature_information
        )
        self.workflow.add_node("get_forecastings", self._workflow_get_forecastings)
        self.workflow.add_node(
            "get_feature_timeseries", self._workflow_get_feature_timeseries
        )
        self.workflow.add_node("get_shap_scores", self._workflow_get_shap_scores)
        self.workflow.add_node("get_llm_insight", self._workflow_get_llm_insight)
        logger.info("All nodes added")

        self.workflow.set_entry_point("get_feature_information")
        logger.info("Entry point added")

        self.workflow.add_edge("get_feature_information", "get_feature_timeseries")
        self.workflow.add_edge("get_feature_timeseries", "get_forecastings")
        self.workflow.add_edge("get_forecastings", "get_shap_scores")
        self.workflow.add_edge("get_shap_scores", "get_llm_insight")
        self.workflow.add_edge("get_llm_insight", END)
        logger.info("All edges added")

        self.app = self.workflow.compile()
        logger.info("LangGraph workflow compiled")

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
        logger.info(f"Found {len(artifacts)} forecast model artifacts")
        self.models = [
            AutonomusForecastModelArchitecture.load(path=path) for path in artifacts
        ]
        logger.info("All forecast models loaded")

    def _load_background_data_shap(self):
        self.bg_data = np.stack(list(self.x.values())).astype(np.float32)
        idx = np.random.choice(len(self.bg_data), 100, replace=False)
        self.bg_data = self.bg_data[idx]
        logger.info("Background data sampled")

        n_samples, n_steps, n_features = self.bg_data.shape
        self.bg_data = self.scaler_artifact.transform(
            self.bg_data.reshape(-1, n_features)
        )
        self.bg_data = self.bg_data.reshape(n_samples, n_steps, n_features)
        logger.info("Background data scaled")

    def _workflow_get_feature_information(self, state: StructuredAnalystState):
        logger.info("Initializing Node: get_feature_information")

        meta_data = self.feature_interpretation[state.feature_name]

        state.feature_business_name = meta_data["business_name"]
        state.feature_macro_role = meta_data["macro_role"]
        state.feature_asset_class = meta_data["asset_class"]
        state.feature_relation_to_lme_al = meta_data["relationship_to_aluminum"]

        logger.info("Node execution completed")

        return state

    def _workflow_get_forecastings(self, state: StructuredAnalystState):
        logger.info("Initializing Node: get_forecastings")

        self.sub_x = self.x[state.ssd_date]
        self.sub_x = self.scaler_artifact.transform(self.sub_x).reshape(
            1, *self.sub_x.shape
        )
        logger.info("infrerence data filtered and scaled")

        forecasts = [round(model.predict(self.sub_x)[0][0], 2) for model in self.models]
        logger.info(f"Return value forecasted - {forecasts}")

        forecasts = [
            round(((self.current_spot * return_val) + self.current_spot))
            for return_val in forecasts
        ]
        self.forecasts = pd.DataFrame(
            {"FutureDates": self.next_dates, "ForcastedPrice": forecasts}
        )
        logger.info("Returns converted to actual prices")

        state.ssd_lme_price = self.current_spot
        state.lme_al_forecast = self.forecasts.to_toon()

        logger.info("Node execution completed")

        return state

    def _workflow_get_feature_timeseries(self, state: StructuredAnalystState):
        logger.info("Initializing Node: get_feature_timeseries")

        self.feature_timeseries = (
            self.raw_features.loc[self.raw_features.ssd <= self.ssd_date]
            .sort_values("ssd", ascending=True)
            .reset_index(drop=True)
            .iloc[-self.bg_data.shape[1] :][["ssd", self.feature_name]]
            .reset_index(drop=True)
            .round({self.feature_name: 2})
        )
        state.feature_timeseries = self.feature_timeseries.to_toon()

        logger.info("Node execution completed")

        return state

    def _workflow_get_shap_scores(self, state: StructuredAnalystState):
        logger.info("Initializing Node: get_shap_scores")

        feature_index = self.feature_order.index(state.feature_name)
        shap_scores = [
            exp.shap_values(self.sub_x)[0][0][:, feature_index]
            for exp in self.shap_explainers
        ]
        columns = [f"SHAPScoreFor {date}" for date in self.next_dates]
        shap_scores = pd.DataFrame(zip(columns, shap_scores))
        shap_scores["ssd"] = self.feature_timeseries["ssd"]
        shap_scores = shap_scores.round(4)
        shap_scores.columns = [str(col) for col in shap_scores.columns]
        logger.info("SHAP scores generated and extracted")

        state.shap_score = shap_scores.to_toon()

        logger.info("Node execution completed")

        return state

    def _workflow_get_llm_insight(self, state: StructuredAnalystState):
        logger.info("Initializing Node: get_llm_insight")

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

        self.messages = [
            ("system", StructuredSystemPrompt.prompt),
            ("user", user_prompt + "\n\n" + self.parser.get_format_instructions()),
        ]

        response = self.llm.invoke(self.messages)

        try:
            response_dict = json.loads(response.content)
            state.insight = StructuredInsight(**response_dict)
        except:
            state.insight = self.parser.parse(response.content)

        logger.info("Node execution completed")
        return state

    def get_insights(self):
        logger.info(
            f"Executing analyst workflow | date={self.ssd_date} | feature={self.feature_name}"
        )

        response = self.app.invoke(
            {"ssd_date": self.ssd_date, "feature_name": self.feature_name}
        )

        logger.info("Workflow execution finished")

        return response
