"""Endpoints for current global LightGBM volatility forecasts."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_prediction_service
from src.schemas.prediction import ModelPredictionResponse, PredictionResponse
from src.services.prediction_service import (
    FORECAST_HORIZON_TRADING_DAYS,
    InsufficientHistoryError,
    LightGBMVolatilityPredictionService,
    MissingMarketContextError,
    PredictionServiceError,
    PredictionUnavailableError,
    TARGET_DESCRIPTION,
    TickerNotFoundError,
    UnsupportedTickerError,
)

router = APIRouter(prefix="/v1", tags=["predictions"])


@router.post("/predictions/{ticker}", response_model=PredictionResponse)
def predict_latest_volatility(
    ticker: str,
    service: LightGBMVolatilityPredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Forecast annualized realized volatility for the next 20 trading days."""

    try:
        forecast = service.predict_latest(ticker)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (UnsupportedTickerError, InsufficientHistoryError, MissingMarketContextError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except (PredictionUnavailableError, PredictionServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return PredictionResponse(
        ticker=forecast.ticker,
        company_name=forecast.company_name,
        as_of_date=forecast.as_of_date,
        target_description=TARGET_DESCRIPTION,
        forecast_horizon_trading_days=FORECAST_HORIZON_TRADING_DAYS,
        champion_model_id=forecast.champion_model_id,
        predictions=[
            ModelPredictionResponse(
                model_id=item.model.name,
                display_name=item.model.display_name,
                model_version=item.model.version,
                predicted_rv_20d=item.predicted_rv_20d,
            )
            for item in forecast.predictions
        ],
    )
