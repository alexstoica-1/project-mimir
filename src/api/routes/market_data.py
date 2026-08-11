"""Endpoints exposing causal engineered market history for the dashboard."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_prediction_service
from src.schemas.prediction import (
    MarketHistoryPointResponse,
    MarketHistoryResponse,
    MarketSummaryResponse,
)
from src.services.prediction_service import (
    InsufficientHistoryError,
    LightGBMVolatilityPredictionService,
    MissingMarketContextError,
    PredictionServiceError,
    TickerNotFoundError,
    UnsupportedTickerError,
)

router = APIRouter(prefix="/v1", tags=["market data"])
HistoryRange = Annotated[Literal["3m", "6m", "1y"], Query(alias="range")]


@router.get("/market-summary/{ticker}", response_model=MarketSummaryResponse)
def get_market_summary(
    ticker: str,
    service: LightGBMVolatilityPredictionService = Depends(get_prediction_service),
) -> MarketSummaryResponse:
    """Return latest causal market indicators for one supported ticker."""

    try:
        summary = service.get_market_summary(ticker)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (UnsupportedTickerError, InsufficientHistoryError, MissingMarketContextError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except PredictionServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return MarketSummaryResponse(**summary.__dict__)


@router.get("/market-data/{ticker}", response_model=MarketHistoryResponse)
def get_market_data(
    ticker: str,
    history_range: HistoryRange = "1y",
    service: LightGBMVolatilityPredictionService = Depends(get_prediction_service),
) -> MarketHistoryResponse:
    """Return recent complete causal feature rows for one supported ticker."""

    try:
        history = service.get_market_history(ticker, history_range)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (UnsupportedTickerError, InsufficientHistoryError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except PredictionServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return MarketHistoryResponse(
        ticker=history.ticker,
        range=history.history_range,
        available_observations=history.available_observations,
        points=[MarketHistoryPointResponse(**point.__dict__) for point in history.points],
    )
