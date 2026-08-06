"""Database-backed, reproducible feature engineering for daily market prices."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.features.financial_features import (
    DRAWDOWN_FEATURE_COLUMNS,
    RETURN_FEATURE_COLUMNS,
    VOLATILITY_FEATURE_COLUMNS,
    add_drawdown_feature,
    add_return_features,
    add_volatility_features,
)
from src.features.market_features import MARKET_FEATURE_COLUMNS, add_market_features

logger = logging.getLogger(__name__)

BASE_PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "source",
]
FEATURE_COLUMNS = (
    RETURN_FEATURE_COLUMNS
    + VOLATILITY_FEATURE_COLUMNS
    + MARKET_FEATURE_COLUMNS
    + DRAWDOWN_FEATURE_COLUMNS
)
OUTPUT_COLUMNS = BASE_PRICE_COLUMNS + FEATURE_COLUMNS


class PriceHistoryReader(Protocol):
    """Read contract required by :class:`MarketFeaturePipeline`."""

    def list_available_companies(self, source: str = "yfinance") -> list[str]: ...

    def get_price_history_frame(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame: ...


class MarketFeaturePipeline:
    """Build causal volatility features from one persisted ticker at a time."""

    def __init__(self, repository: PriceHistoryReader) -> None:
        """Create a feature pipeline with an injected market-data reader."""

        self.repository = repository

    def build_for_ticker(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Read and transform one ticker's historical prices."""

        prices = self.repository.get_price_history_frame(
            ticker,
            start=start,
            end=end,
            source=source,
        )
        features = self.transform_price_history(prices)
        logger.info("Built %s feature rows for ticker=%s", len(features), ticker.strip().upper())
        return features

    def build_for_universe(
        self,
        tickers: Iterable[str] | None = None,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Build and combine independently calculated feature frames by ticker."""

        requested_tickers = list(tickers) if tickers is not None else self.repository.list_available_companies(source)
        frames = [
            self.build_for_ticker(ticker, start=start, end=end, source=source)
            for ticker in requested_tickers
        ]
        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return self.empty_output_frame()

        result = pd.concat(non_empty_frames, ignore_index=True)
        return result.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    @classmethod
    def transform_price_history(cls, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the complete feature set from one normalized price frame.

        The source frame is copied before conversion and computation. Rows that
        lack the full required lookback are dropped only from the output frame.
        """

        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")

        missing_columns = sorted(set(BASE_PRICE_COLUMNS) - set(prices.columns))
        if missing_columns:
            raise ValueError(f"prices is missing required columns: {', '.join(missing_columns)}")

        if prices.empty:
            return cls.empty_output_frame()

        result = prices.loc[:, BASE_PRICE_COLUMNS].copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise")
        for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
            result[column] = pd.to_numeric(result[column], errors="raise")

        result = result.sort_values("date", kind="stable").reset_index(drop=True)
        result = add_return_features(result)
        result = add_volatility_features(result)
        result = add_market_features(result)
        result = add_drawdown_feature(result)
        result = result.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        return result.loc[:, OUTPUT_COLUMNS]

    @staticmethod
    def empty_output_frame() -> pd.DataFrame:
        """Return an empty frame with the documented model-ready schema."""

        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    @staticmethod
    def save_csv(features: pd.DataFrame, output_path: Path) -> Path:
        """Save a deterministic model-ready feature dataset and return its path."""

        missing_columns = sorted(set(OUTPUT_COLUMNS) - set(features.columns))
        if missing_columns:
            raise ValueError(f"features is missing output columns: {', '.join(missing_columns)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_features = features.loc[:, OUTPUT_COLUMNS].copy()
        ordered_features.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        logger.info("Saved %s feature rows to %s", len(ordered_features), output_path)
        return output_path
