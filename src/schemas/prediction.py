"""Pydantic response schemas for deployed volatility forecasts."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Identity and training metadata for the model currently served."""

    model_name: str
    model_version: str
    artifact_name: str
    supported_tickers: list[str]
    feature_count: int = Field(ge=1)
    forecast_horizon_trading_days: int = Field(ge=1)
    training_parameters: dict[str, Any]


class PredictionResponse(BaseModel):
    """Latest-date forecast returned by the prediction API."""

    ticker: str
    as_of_date: date
    predicted_rv_20d: float = Field(ge=0)
    unit: Literal["annualized_decimal_volatility"] = "annualized_decimal_volatility"
    forecast_horizon_trading_days: int = Field(ge=1)
    model_name: str
    model_version: str


class MarketHistoryPointResponse(BaseModel):
    """One causal market-data row shown in the dashboard history view."""

    date: date
    adjusted_close: float
    rv_20d: float = Field(ge=0)
    return_20d: float
    drawdown: float
    volume_ratio_20d: float = Field(ge=0)


class MarketHistoryResponse(BaseModel):
    """Recent engineered market history for one ticker and requested range."""

    ticker: str
    range: Literal["3m", "6m", "1y"]
    available_observations: int = Field(ge=1)
    points: list[MarketHistoryPointResponse]


class MarketSummaryResponse(BaseModel):
    """Latest causal market indicators for one supported ticker."""

    ticker: str
    as_of_date: date
    adjusted_close: float
    return_1d: float
    return_5d: float
    return_20d: float
    rv_20d: float = Field(ge=0)
    drawdown: float
    volume_ratio_20d: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Readiness response for the API, database, and model artifact."""

    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"
    model_name: str
    model_version: str
