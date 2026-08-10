"""Endpoints describing the currently deployed forecasting model."""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_model_metadata
from src.schemas.prediction import ModelInfoResponse
from src.services.prediction_service import FORECAST_HORIZON_TRADING_DAYS, ModelMetadata

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def get_deployed_model(metadata: ModelMetadata = Depends(get_model_metadata)) -> ModelInfoResponse:
    """Describe the single global LightGBM artifact loaded by this API instance."""

    return ModelInfoResponse(
        model_name=metadata.name,
        model_version=metadata.version,
        artifact_name=metadata.artifact_name,
        supported_tickers=list(metadata.supported_tickers),
        feature_count=metadata.feature_count,
        forecast_horizon_trading_days=FORECAST_HORIZON_TRADING_DAYS,
        training_parameters=metadata.training_parameters,
    )
