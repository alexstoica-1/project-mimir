"""Endpoints describing the currently deployed forecasting model."""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_served_models_metadata
from src.schemas.prediction import ModelCatalogResponse, ModelDetailResponse
from src.services.prediction_service import (
    CHAMPION_MODEL_ID,
    FORECAST_HORIZON_TRADING_DAYS,
    TARGET_DESCRIPTION,
    ModelMetadata,
)

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model", response_model=ModelCatalogResponse)
def get_deployed_models(
    metadata: tuple[ModelMetadata, ModelMetadata] = Depends(get_served_models_metadata),
) -> ModelCatalogResponse:
    """Describe the compatible LightGBM champion and LSTM comparison model."""

    return ModelCatalogResponse(
        champion_model_id=CHAMPION_MODEL_ID,
        models=[
            ModelDetailResponse(
                model_id=item.name,
                display_name=item.display_name,
                model_version=item.version,
                artifact_name=item.artifact_name,
                supported_tickers=list(item.supported_tickers),
                feature_count=item.feature_count,
                target_description=TARGET_DESCRIPTION,
                forecast_horizon_trading_days=FORECAST_HORIZON_TRADING_DAYS,
                input_requirement=item.input_requirement,
                lookback_observations=item.lookback_observations,
                training_parameters=item.training_parameters,
            )
            for item in metadata
        ],
    )
