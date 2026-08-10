"""Database-backed, reproducible stock feature engineering pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.features.financial_features import (
    DISTRIBUTION_FEATURE_COLUMNS,
    DRAWDOWN_FEATURE_COLUMNS,
    RANGE_FEATURE_COLUMNS,
    REALIZED_VOLATILITY_FEATURE_COLUMNS,
    RETURN_FEATURE_COLUMNS,
    TARGET_COLUMN,
    VOLUME_FEATURE_COLUMNS,
    add_distribution_features,
    add_drawdown_features,
    add_range_features,
    add_realized_volatility_features,
    add_return_features,
    add_volume_features,
    create_future_rv_target,
)
from src.features.market_features import (
    CONTEXT_FEATURE_COLUMNS,
    SPY_TICKER,
    VIX_TICKER,
    add_market_correlation,
    build_spy_features,
    build_vix_features,
    merge_context_features,
)

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

# Keep the public schema in the requested modeling order, independent of the
# internal function grouping used to construct each feature family.
FEATURE_COLUMNS = [
    "log_return",
    "abs_log_return",
    "log_return_squared",
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "rv_20d_lag1",
    "rv_20d_lag5",
    "rv_20d_lag20",
    "rv_change_1d",
    "vol_of_vol_20d",
    "log_high_low",
    "log_close_open",
    "overnight_gap",
    "return_5d",
    "return_20d",
    "return_60d",
    "drawdown",
    "max_drawdown_20d",
    "max_drawdown_60d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_volatility_20d",
    "rolling_skew_20d",
    "rolling_kurtosis_20d",
    "market_return_20d",
    "market_rv_20d",
    "market_corr_60d",
    "vix",
    "vix_change_5d",
    "vix_minus_market_rv",
]
STOCK_FEATURE_COLUMNS = (
    RETURN_FEATURE_COLUMNS
    + REALIZED_VOLATILITY_FEATURE_COLUMNS
    + RANGE_FEATURE_COLUMNS
    + DRAWDOWN_FEATURE_COLUMNS
    + VOLUME_FEATURE_COLUMNS
    + DISTRIBUTION_FEATURE_COLUMNS
)
MODEL_REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
OUTPUT_COLUMNS = BASE_PRICE_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
INFERENCE_OUTPUT_COLUMNS = BASE_PRICE_COLUMNS + FEATURE_COLUMNS

if set(FEATURE_COLUMNS) != set(STOCK_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS):
    raise RuntimeError("Feature-column declarations are inconsistent")


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


def validate_and_sort_raw_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized long-format prices and sort by ticker/date."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    missing_columns = sorted(set(BASE_PRICE_COLUMNS) - set(prices.columns))
    if missing_columns:
        raise ValueError(f"prices is missing required columns: {', '.join(missing_columns)}")

    result = prices.loc[:, BASE_PRICE_COLUMNS].copy()
    if result.empty:
        return result

    if result["ticker"].isna().any() or result["ticker"].astype(str).str.strip().eq("").any():
        raise ValueError("ticker must be present on every raw price row")
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError("date must be present on every raw price row")

    for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
        result[column] = pd.to_numeric(result[column], errors="raise")

    duplicates = result.duplicated(subset=["ticker", "date"], keep=False)
    if duplicates.any():
        duplicate_keys = result.loc[duplicates, ["ticker", "date"]].drop_duplicates()
        raise ValueError(
            "raw prices contains duplicate ticker/date rows: "
            f"{duplicate_keys.to_dict('records')}"
        )
    return result.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def split_asset_roles(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate prediction stocks, SPY benchmark rows, and VIX context rows."""

    stocks = prices.loc[~prices["ticker"].isin([SPY_TICKER, VIX_TICKER])].copy()
    spy = prices.loc[prices["ticker"].eq(SPY_TICKER)].copy()
    vix = prices.loc[prices["ticker"].eq(VIX_TICKER)].copy()
    return stocks, spy, vix


def add_stock_features(stocks: pd.DataFrame) -> pd.DataFrame:
    """Apply all causal stock-specific feature families in sequence."""

    result = add_return_features(stocks)
    result = add_realized_volatility_features(result)
    result = add_range_features(result)
    result = add_drawdown_features(result)
    result = add_volume_features(result)
    return add_distribution_features(result)


def finalize_modeling_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Drop only rows lacking a full feature history or forward target.

    Financially meaningful missing values remain NaN throughout construction.
    This final model-ready boundary removes rows with unavailable inputs or
    labels; it never fills them with zero or forward-fills context values.
    """

    result = features.replace([np.inf, -np.inf], np.nan)
    missing_context = result[["market_rv_20d", "vix"]].isna().sum().to_dict()
    if any(missing_context.values()):
        logger.info("Exact-date context missingness before final filtering: %s", missing_context)

    result = result.dropna(subset=MODEL_REQUIRED_COLUMNS)
    result = result.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    if result["ticker"].isin([SPY_TICKER, VIX_TICKER]).any():
        raise ValueError("context assets cannot appear as prediction rows")
    if result.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("final dataset contains duplicate ticker/date rows")
    return result.loc[:, OUTPUT_COLUMNS]


def finalize_inference_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with complete causal inputs, without requiring a future target.

    The target is deliberately unavailable on the newest observations.  Serving
    must therefore apply the same feature-completeness rule as training while
    omitting ``target_rv_20d`` from its output.
    """

    result = features.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=FEATURE_COLUMNS)
    result = result.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    if result["ticker"].isin([SPY_TICKER, VIX_TICKER]).any():
        raise ValueError("context assets cannot appear as prediction rows")
    if result.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("final inference dataset contains duplicate ticker/date rows")
    return result.loc[:, INFERENCE_OUTPUT_COLUMNS]


class MarketFeaturePipeline:
    """Build stock prediction rows enriched with date-aligned SPY/VIX context."""

    def __init__(self, repository: PriceHistoryReader) -> None:
        """Create a feature pipeline with an injected market-data reader."""

        self.repository = repository

    def load_price_data(
        self,
        tickers: Iterable[str],
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Load requested tickers through the existing repository contract."""

        normalized_tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        frames = [
            self.repository.get_price_history_frame(
                ticker,
                start=start,
                end=end,
                source=source,
            )
            for ticker in normalized_tickers
        ]
        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return pd.DataFrame(columns=BASE_PRICE_COLUMNS)
        return pd.concat(non_empty_frames, ignore_index=True)

    def build_for_ticker(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
        inference_ready: bool = False,
    ) -> pd.DataFrame:
        """Build one prediction ticker with SPY and VIX context loaded alongside it."""

        normalized_ticker = ticker.strip().upper()
        if normalized_ticker in {SPY_TICKER, VIX_TICKER}:
            raise ValueError(f"{normalized_ticker} is a context asset, not a prediction ticker")
        features = self.build_for_universe(
            [normalized_ticker],
            start=start,
            end=end,
            source=source,
            inference_ready=inference_ready,
        )
        logger.info("Built %s feature rows for ticker=%s", len(features), normalized_ticker)
        return features

    def build_for_universe(
        self,
        tickers: Iterable[str] | None = None,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
        inference_ready: bool = False,
    ) -> pd.DataFrame:
        """Build prediction assets together, using SPY/VIX only as context."""

        available = (
            list(tickers)
            if tickers is not None
            else self.repository.list_available_companies(source)
        )
        prediction_tickers = sorted(
            {
                ticker.strip().upper()
                for ticker in available
                if ticker.strip().upper() not in {SPY_TICKER, VIX_TICKER}
            }
        )
        if not prediction_tickers:
            return self.empty_inference_frame() if inference_ready else self.empty_output_frame()

        raw_prices = self.load_price_data(
            [*prediction_tickers, SPY_TICKER, VIX_TICKER],
            start=start,
            end=end,
            source=source,
        )
        return self.transform_price_history(raw_prices, inference_ready=inference_ready)

    @classmethod
    def transform_price_history(
        cls,
        prices: pd.DataFrame,
        *,
        model_ready: bool = True,
        inference_ready: bool = False,
    ) -> pd.DataFrame:
        """Transform normalized long-format stock, SPY, and VIX price history.

        With ``model_ready=False``, rolling/context/target NaNs are preserved for
        inspection. With ``inference_ready=True``, rows require complete causal
        features but not a future target, so the most recent usable row remains.
        The default applies the training-data filter.
        """

        normalized = validate_and_sort_raw_data(prices)
        if normalized.empty:
            return cls.empty_inference_frame() if inference_ready else cls.empty_output_frame()

        stocks, spy_prices, vix_prices = split_asset_roles(normalized)
        if stocks.empty:
            return cls.empty_inference_frame() if inference_ready else cls.empty_output_frame()

        stock_features = add_stock_features(stocks)
        spy_features = build_spy_features(spy_prices)
        vix_features = build_vix_features(vix_prices)
        stock_features = add_market_correlation(stock_features, spy_features)
        merged = merge_context_features(stock_features, spy_features, vix_features)
        with_target = create_future_rv_target(merged)
        with_target = with_target.sort_values(["ticker", "date"], kind="stable").reset_index(
            drop=True
        )
        if inference_ready:
            return finalize_inference_dataset(with_target)
        if model_ready:
            return finalize_modeling_dataset(with_target)
        return with_target.loc[:, OUTPUT_COLUMNS]

    @staticmethod
    def empty_output_frame() -> pd.DataFrame:
        """Return an empty frame with the documented model-ready schema."""

        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    @staticmethod
    def empty_inference_frame() -> pd.DataFrame:
        """Return an empty frame matching the inference feature schema."""

        return pd.DataFrame(columns=INFERENCE_OUTPUT_COLUMNS)

    @staticmethod
    def save_csv(features: pd.DataFrame, output_path: Path) -> Path:
        """Save a deterministic model-ready feature dataset and return its path."""

        missing_columns = sorted(set(OUTPUT_COLUMNS) - set(features.columns))
        if missing_columns:
            raise ValueError(f"features is missing output columns: {', '.join(missing_columns)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_features = features.loc[:, OUTPUT_COLUMNS].copy()
        ordered_features = ordered_features.sort_values(
            ["ticker", "date"], kind="stable"
        ).reset_index(drop=True)
        ordered_features.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        logger.info("Saved %s feature rows to %s", len(ordered_features), output_path)
        return output_path
