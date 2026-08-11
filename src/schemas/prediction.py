"""Pydantic response schemas for deployed volatility forecasts."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelDetailResponse(BaseModel):
    """Identity and inference requirements for one served forecasting model."""

    model_id: str
    display_name: str
    model_version: str
    artifact_name: str
    supported_tickers: list[str]
    feature_count: int = Field(ge=1)
    target_name: Literal["target_rv_20d"] = "target_rv_20d"
    target_description: str
    forecast_horizon_trading_days: int = Field(ge=1)
    input_requirement: str
    lookback_observations: int | None = Field(default=None, ge=1)
    training_parameters: dict[str, Any]


class ModelCatalogResponse(BaseModel):
    """The two compatible models loaded by this API instance."""

    champion_model_id: str
    models: list[ModelDetailResponse]


class ModelPredictionResponse(BaseModel):
    """One model's latest-date volatility forecast."""

    model_id: str
    display_name: str
    model_version: str
    predicted_rv_20d: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Latest-date volatility forecasts from every served model."""

    ticker: str
    company_name: str
    as_of_date: date
    target_name: Literal["target_rv_20d"] = "target_rv_20d"
    target_description: str
    unit: Literal["annualized_decimal_volatility"] = "annualized_decimal_volatility"
    forecast_horizon_trading_days: int = Field(ge=1)
    champion_model_id: str
    predictions: list[ModelPredictionResponse]


class MarketHistoryPointResponse(BaseModel):
    """One causal market-data row shown in the dashboard history view."""

    date: date
    adjusted_close: float
    rv_20d: float = Field(ge=0)
    return_5d: float
    return_20d: float
    drawdown: float
    volume_ratio_20d: float = Field(ge=0)


class MarketHistoryResponse(BaseModel):
    """Recent engineered market history for one ticker and requested range."""

    ticker: str
    range: Literal["1m", "3m", "6m", "1y", "5y"]
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
    served_models: list[str]
