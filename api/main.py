import os
from pathlib import Path
from datetime import datetime
from typing import Final, List
from pydantic import BaseModel
from dotenv import load_dotenv
from core.logging import LoggerFactory
from fastapi import FastAPI, HTTPException
from core.graph import StructuredFeatureMarketAnalyst

logger = LoggerFactory().get_logger(__name__)
logger.info("Initializing Autonomous Metal API service...")


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from %s", ENV_PATH)

GROQ_MODEL = os.getenv("GROQ_LLM_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FORECAST_HORIZON = os.getenv("FORECAST_HORIZON")
START_DATE: Final[datetime] = datetime(2015, 1, 14)
END_DATE: Final[datetime] = datetime(2026, 2, 5)

if not GROQ_MODEL or not GROQ_API_KEY:
    logger.error("Missing GROQ configuration in environment variables.")
    raise RuntimeError("GROQ configuration not found in .env")

if FORECAST_HORIZON is None:
    logger.error("FORECAST_HORIZON not defined in environment.")
    raise RuntimeError("FORECAST_HORIZON missing")

forecast_horizon: int = int(FORECAST_HORIZON)

logger.info("Configuration loaded successfully | horizon=%s", forecast_horizon)

ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"

scaler_artifact_path: Final[Path] = ARTIFACTS_DIR / "feature-scaler.pkl"
forecast_models_artifact_directory_path: Final[Path] = ARTIFACTS_DIR
feature_order_path: Final[Path] = ARTIFACTS_DIR / "features-set.pkl"
feature_interpretation_path: Final[Path] = ARTIFACTS_DIR / "feature-interpretation.json"
spot_price_path: Final[Path] = ARTIFACTS_DIR / "spot-prices.csv"
x_data_path: Final[Path] = ARTIFACTS_DIR / "all-x.pkl"
raw_features_path: Final[Path] = ARTIFACTS_DIR / "features.csv"

logger.info("Artifact directory configured at %s", ARTIFACTS_DIR)

drivers_list: List[str] = [
    "lme_al",
    "bhp_group_stocks",
    "brent_crude",
    "china_a450",
    "cobe_volatility",
    "dax_performance",
    "ftse100",
    "hindalco_stocks",
    "hsi",
    "rio_tinto_stocks",
    "snp500",
    "spgsia",
    "lme_al_inventory",
    "baltic_dry_index",
]

logger.info("Loaded %d feature drivers", len(drivers_list))


app = FastAPI(
    title="Autonomous Metal API",
    description="Generate LME Aluminum analyst reports",
    version="1.0",
)

logger.info("FastAPI application initialized")


class ReportRequest(BaseModel):
    friday_date: str


def validate_friday(date_str: str) -> datetime:
    logger.info("Validating input date: %s", date_str)

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning("Invalid date format received: %s", date_str)
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    # Friday check
    if date_obj.weekday() != 4:
        logger.warning("Non-Friday date rejected: %s", date_str)
        raise HTTPException(status_code=400, detail="Provided date is not a Friday")

    # Range check
    if not (START_DATE <= date_obj <= END_DATE):
        logger.warning("Date outside allowed range: %s", date_str)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Date must be between "
                f"{START_DATE.strftime('%Y-%m-%d')} and "
                f"{END_DATE.strftime('%Y-%m-%d')}"
            ),
        )

    return date_obj

    return date_obj


@app.post(
    "/generate-report",
    summary="Generate weekly LME Aluminum analyst report",
    description=(
        "Generates a markdown-formatted analyst report for a given Friday.\n\n"
        "Constraints:\n"
        "- Date must be a **Friday**.\n"
        f"- Date must be between **{START_DATE.strftime('%Y-%m-%d')}** "
        f"and **{END_DATE.strftime('%Y-%m-%d')}**."
    ),
)
def create_report(request: ReportRequest):

    logger.info("Report generation request received | date=%s", request.friday_date)

    validate_friday(request.friday_date)

    try:
        logger.info("Initializing StructuredFeatureMarketAnalyst")

        analyst = StructuredFeatureMarketAnalyst(
            groq_model=GROQ_MODEL,
            groq_api=GROQ_API_KEY,
            scaler_artifact_path=scaler_artifact_path,
            forecast_models_artifact_directory_path=forecast_models_artifact_directory_path,
            feature_order_path=feature_order_path,
            feature_interpretation_path=feature_interpretation_path,
            spot_price_path=spot_price_path,
            x_data_path=x_data_path,
            raw_features_path=raw_features_path,
            ssd_date=request.friday_date,
            feature_list=drivers_list,
            horizon=forecast_horizon,
        )

        logger.info("Running analyst pipeline...")
        markdown_report = analyst.get_analyst_report_on_structured_inputs()

        logger.info("Report successfully generated | date=%s", request.friday_date)

        return {
            "status": "success",
            "report_date": request.friday_date,
            "markdown_report": markdown_report,
        }

    except Exception as e:
        logger.exception(
            "Report generation failed | date=%s | error=%s", request.friday_date, str(e)
        )

        raise HTTPException(status_code=500, detail="Report generation failed")
