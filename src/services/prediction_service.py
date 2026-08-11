"""Application service for serving the deployed global LightGBM model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS, MarketFeaturePipeline
from src.ml.lightgbm_model import LightGBMRegressorModel
from src.ml.lstm_model import LSTMTrainingResult

MINIMUM_PRICE_OBSERVATIONS = 61
FORECAST_HORIZON_TRADING_DAYS = 20
CHAMPION_MODEL_ID = "lightgbm-global"
TARGET_NAME = "target_rv_20d"
TARGET_DESCRIPTION = "20-trading-day annualized realized volatility"
MARKET_HISTORY_RANGES = {"3m": 63, "6m": 126, "1y": 252}
MARKET_HISTORY_COLUMNS = [
    "date",
    "adjusted_close",
    "rv_20d",
    "return_20d",
    "drawdown",
    "volume_ratio_20d",
]


class PredictionServiceError(Exception):
    """Base class for expected prediction-service failures."""


class TickerNotFoundError(PredictionServiceError):
    """Raised when no persisted market data exists for a ticker."""


class UnsupportedTickerError(PredictionServiceError):
    """Raised when a ticker was not present in model training."""


class InsufficientHistoryError(PredictionServiceError):
    """Raised when an asset does not have enough price history for features."""


class MissingMarketContextError(PredictionServiceError):
    """Raised when the latest asset date lacks SPY or VIX feature context."""


class PredictionUnavailableError(PredictionServiceError):
    """Raised when an artifact cannot produce a valid forecast."""


@dataclass(frozen=True)
class ModelMetadata:
    """Stable, response-safe metadata for the deployed artifact."""

    name: str
    display_name: str
    version: str
    artifact_name: str
    supported_tickers: tuple[str, ...]
    feature_count: int
    input_requirement: str
    lookback_observations: int | None
    training_parameters: dict[str, object]


@dataclass(frozen=True)
class VolatilityPrediction:
    """Latest-date forecasts from every model served by this API."""

    ticker: str
    as_of_date: date
    champion_model_id: str
    predictions: tuple["ModelForecast", ...]


@dataclass(frozen=True)
class ModelForecast:
    """One model's latest-date annualized volatility forecast."""

    predicted_rv_20d: float
    model: ModelMetadata


@dataclass(frozen=True)
class MarketHistoryPoint:
    """One causal engineered market-data observation for dashboard display."""

    date: date
    adjusted_close: float
    rv_20d: float
    return_20d: float
    drawdown: float
    volume_ratio_20d: float


@dataclass(frozen=True)
class MarketHistory:
    """A recent, ticker-specific slice of complete causal feature rows."""

    ticker: str
    history_range: str
    available_observations: int
    points: tuple[MarketHistoryPoint, ...]


@dataclass(frozen=True)
class MarketSummary:
    """Latest causal market indicators for one dashboard ticker."""

    ticker: str
    as_of_date: date
    adjusted_close: float
    return_1d: float
    return_5d: float
    return_20d: float
    rv_20d: float
    drawdown: float
    volume_ratio_20d: float


def build_model_metadata(
    model: LightGBMRegressorModel,
    artifact_path: str | Path,
) -> ModelMetadata:
    """Build metadata from a loaded global LightGBM artifact."""

    if model.preprocessor is None:
        raise PredictionUnavailableError("Deployed LightGBM artifact has no fitted preprocessor")
    path = Path(artifact_path)
    if not path.is_file():
        raise PredictionUnavailableError(f"Model artifact does not exist: {path}")
    tickers = tuple(sorted(str(value) for value in model.preprocessor.ticker_encoder.categories_[0]))
    return ModelMetadata(
        name="lightgbm-global",
        display_name="LightGBM",
        version=sha256(path.read_bytes()).hexdigest()[:12],
        artifact_name=path.name,
        supported_tickers=tickers,
        feature_count=len(FEATURE_COLUMNS),
        input_requirement="Latest causal feature row",
        lookback_observations=None,
        training_parameters=dict(model.params),
    )


def build_lstm_model_metadata(
    model: LSTMTrainingResult,
    artifact_path: str | Path,
) -> ModelMetadata:
    """Build response-safe metadata from a saved global LSTM artifact."""

    path = Path(artifact_path)
    if not path.is_file():
        raise PredictionUnavailableError(f"LSTM artifact does not exist: {path}")
    tickers = tuple(sorted(str(value) for value in model.preprocessor.ticker_encoder.categories_[0]))
    return ModelMetadata(
        name="lstm-global",
        display_name="LSTM",
        version=sha256(path.read_bytes()).hexdigest()[:12],
        artifact_name=path.name,
        supported_tickers=tickers,
        feature_count=len(model.preprocessor.feature_columns),
        input_requirement=f"Latest {model.lookback} causal feature rows",
        lookback_observations=model.lookback,
        training_parameters={
            "hidden_size": model.model.hidden_size,
            "num_layers": model.model.num_layers,
            "dropout": model.model.dropout,
        },
    )


def validate_served_model_compatibility(
    lightgbm_model: LightGBMRegressorModel,
    lstm_model: LSTMTrainingResult,
) -> None:
    """Require both served models to use the same ticker and feature universe."""

    if lightgbm_model.preprocessor is None:
        raise PredictionUnavailableError("Deployed LightGBM artifact has no fitted preprocessor")
    if list(lightgbm_model.preprocessor.feature_columns) != list(lstm_model.preprocessor.feature_columns):
        raise PredictionUnavailableError("LightGBM and LSTM artifacts use different feature columns")
    lightgbm_tickers = set(lightgbm_model.preprocessor.ticker_encoder.categories_[0])
    lstm_tickers = set(lstm_model.preprocessor.ticker_encoder.categories_[0])
    if lightgbm_tickers != lstm_tickers:
        raise PredictionUnavailableError("LightGBM and LSTM artifacts support different ticker universes")


class LightGBMVolatilityPredictionService:
    """Build current causal features from persisted prices and predict volatility."""

    def __init__(
        self,
        *,
        model: LightGBMRegressorModel,
        metadata: ModelMetadata,
        lstm_model: LSTMTrainingResult,
        lstm_metadata: ModelMetadata,
        repository: object,
        source: str = "yfinance",
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.lstm_model = lstm_model
        self.lstm_metadata = lstm_metadata
        self.repository = repository
        self.source = source

    def predict_latest(self, ticker: str) -> VolatilityPrediction:
        """Predict the next 20-day realized volatility with both served models."""

        normalized_ticker, features = self._get_usable_inference_features(ticker)
        latest_features = features.iloc[[-1]].copy()
        as_of_date = pd.Timestamp(latest_features["date"].iloc[0]).date()

        lightgbm_prediction = float(self.model.predict(latest_features)[0])
        if not np.isfinite(lightgbm_prediction) or lightgbm_prediction < 0:
            raise PredictionUnavailableError("LightGBM model produced an invalid volatility prediction")
        if len(features) < self.lstm_model.lookback:
            raise InsufficientHistoryError(
                f"Ticker {normalized_ticker} has {len(features)} complete feature rows; "
                f"LSTM requires {self.lstm_model.lookback}"
            )
        try:
            lstm_prediction = self.lstm_model.predict_latest(features)
        except ValueError as exc:
            raise PredictionUnavailableError(f"LSTM could not build a live prediction sequence: {exc}") from exc
        if not np.isfinite(lstm_prediction) or lstm_prediction < 0:
            raise PredictionUnavailableError("LSTM model produced an invalid volatility prediction")
        return VolatilityPrediction(
            ticker=normalized_ticker,
            as_of_date=as_of_date,
            champion_model_id=CHAMPION_MODEL_ID,
            predictions=(
                ModelForecast(predicted_rv_20d=lightgbm_prediction, model=self.metadata),
                ModelForecast(predicted_rv_20d=lstm_prediction, model=self.lstm_metadata),
            ),
        )

    def get_market_summary(self, ticker: str) -> MarketSummary:
        """Return the latest causal indicators used to describe one ticker."""

        normalized_ticker, latest_features = self._get_latest_usable_features(ticker)
        row = latest_features.iloc[0]
        return MarketSummary(
            ticker=normalized_ticker,
            as_of_date=pd.Timestamp(row.date).date(),
            adjusted_close=float(row.adjusted_close),
            return_1d=float(np.expm1(row.log_return)),
            return_5d=float(row.return_5d),
            return_20d=float(row.return_20d),
            rv_20d=float(row.rv_20d),
            drawdown=float(row.drawdown),
            volume_ratio_20d=float(row.volume_ratio_20d),
        )

    def get_market_history(self, ticker: str, history_range: str = "1y") -> MarketHistory:
        """Return recent causal features for one trained ticker's market view."""

        normalized_ticker = self._validate_supported_ticker(ticker)
        requested_observations = MARKET_HISTORY_RANGES.get(history_range)
        if requested_observations is None:
            raise ValueError(f"Unsupported market-data range: {history_range}")

        features = MarketFeaturePipeline(self.repository).build_for_ticker(
            normalized_ticker,
            source=self.source,
            inference_ready=True,
        )
        if len(features) < requested_observations:
            raise InsufficientHistoryError(
                f"Ticker {normalized_ticker} has {len(features)} complete feature rows; "
                f"{history_range} requires {requested_observations}"
            )

        history = features.loc[:, MARKET_HISTORY_COLUMNS].tail(requested_observations)
        points = tuple(
            MarketHistoryPoint(
                date=pd.Timestamp(row.date).date(),
                adjusted_close=float(row.adjusted_close),
                rv_20d=float(row.rv_20d),
                return_20d=float(row.return_20d),
                drawdown=float(row.drawdown),
                volume_ratio_20d=float(row.volume_ratio_20d),
            )
            for row in history.itertuples(index=False)
        )
        return MarketHistory(
            ticker=normalized_ticker,
            history_range=history_range,
            available_observations=len(features),
            points=points,
        )

    def _validate_supported_ticker(self, ticker: str) -> str:
        """Validate persisted data availability and deployed-model support."""

        normalized_ticker = self._normalize_ticker(ticker)
        if not self.repository.company_exists(normalized_ticker):  # type: ignore[attr-defined]
            raise TickerNotFoundError(f"No persisted market data exists for ticker {normalized_ticker}")
        if normalized_ticker not in self.metadata.supported_tickers:
            raise UnsupportedTickerError(
                f"Ticker {normalized_ticker} is not supported by the deployed model"
            )
        return normalized_ticker

    def _get_latest_usable_features(self, ticker: str) -> tuple[str, pd.DataFrame]:
        """Build the latest inference row and require exact-date market context."""

        normalized_ticker, features = self._get_usable_inference_features(ticker)
        return normalized_ticker, features.iloc[[-1]].copy()

    def _get_usable_inference_features(self, ticker: str) -> tuple[str, pd.DataFrame]:
        """Build all complete causal rows and require current market context."""

        normalized_ticker = self._validate_supported_ticker(ticker)
        latest_price_date = self.repository.latest_price_date(normalized_ticker, self.source)  # type: ignore[attr-defined]
        price_history = self.repository.get_price_history_frame(  # type: ignore[attr-defined]
            normalized_ticker,
            source=self.source,
        )
        if latest_price_date is None or price_history.empty:
            raise TickerNotFoundError(f"No persisted {self.source} prices exist for {normalized_ticker}")
        if len(price_history) < MINIMUM_PRICE_OBSERVATIONS:
            raise InsufficientHistoryError(
                f"Ticker {normalized_ticker} needs at least {MINIMUM_PRICE_OBSERVATIONS} price observations"
            )

        features = MarketFeaturePipeline(self.repository).build_for_ticker(
            normalized_ticker,
            source=self.source,
            inference_ready=True,
        )
        if features.empty:
            raise InsufficientHistoryError(
                f"Ticker {normalized_ticker} has no complete causal feature row"
            )

        as_of_date = pd.Timestamp(features["date"].iloc[-1]).date()
        if as_of_date != latest_price_date:
            raise MissingMarketContextError(
                f"Latest {normalized_ticker} price date {latest_price_date} has no matching SPY/VIX context"
            )
        return normalized_ticker, features

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized:
            raise TickerNotFoundError("ticker must be a non-empty string")
        return normalized
