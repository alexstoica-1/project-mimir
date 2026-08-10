"""Causal stock-level feature calculations for normalized daily prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
REALIZED_VOLATILITY_WINDOWS = (5, 10, 20, 60)

RETURN_FEATURE_COLUMNS = [
    "log_return",
    "abs_log_return",
    "log_return_squared",
    "return_5d",
    "return_20d",
    "return_60d",
]
REALIZED_VOLATILITY_FEATURE_COLUMNS = [
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "rv_20d_lag1",
    "rv_20d_lag5",
    "rv_20d_lag20",
    "rv_change_1d",
    "vol_of_vol_20d",
]
RANGE_FEATURE_COLUMNS = ["log_high_low", "log_close_open", "overnight_gap"]
DRAWDOWN_FEATURE_COLUMNS = ["drawdown", "max_drawdown_20d", "max_drawdown_60d"]
VOLUME_FEATURE_COLUMNS = ["volume_ratio_5d", "volume_ratio_20d", "volume_volatility_20d"]
DISTRIBUTION_FEATURE_COLUMNS = ["rolling_skew_20d", "rolling_kurtosis_20d"]
TARGET_COLUMN = "target_rv_20d"


def _positive_log(values: pd.Series) -> pd.Series:
    """Return natural logs for strictly positive values and NaN otherwise."""

    return np.log(values.where(values > 0)) # type: ignore


def _rolling_by_ticker(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
) -> pd.Series:
    """Apply a full-window rolling operation independently within each ticker."""

    grouped = frame.groupby("ticker", sort=False, observed=True)[column]
    return grouped.transform(
        lambda values: getattr(values.rolling(window=window, min_periods=window), operation)()
    )


def add_return_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted-close daily and cumulative log returns within ticker.

    The input must be sorted by ticker/date. Every return is based on the
    current row and an earlier observation from the same ticker.
    """

    result = prices.copy()
    log_adjusted_close = _positive_log(result["adjusted_close"])
    grouped_log_price = log_adjusted_close.groupby(result["ticker"], sort=False, observed=True)

    result["log_return"] = grouped_log_price.diff()
    result["abs_log_return"] = result["log_return"].abs()
    result["log_return_squared"] = result["log_return"].pow(2)
    for window in (5, 20, 60):
        result[f"return_{window}d"] = grouped_log_price.diff(periods=window)
    return result


def add_realized_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add annualized trailing realized volatility and its causal derivatives."""

    result = prices.copy()
    for window in REALIZED_VOLATILITY_WINDOWS:
        mean_squared_return = _rolling_by_ticker(
            result,
            "log_return_squared",
            window,
            "mean",
        )
        result[f"rv_{window}d"] = np.sqrt(TRADING_DAYS_PER_YEAR * mean_squared_return)

    rv_grouped = result.groupby("ticker", sort=False, observed=True)["rv_20d"]
    for lag in (1, 5, 20):
        result[f"rv_20d_lag{lag}"] = rv_grouped.shift(lag)
    result["rv_change_1d"] = rv_grouped.diff()
    result["vol_of_vol_20d"] = _rolling_by_ticker(result, "rv_20d", 20, "std")
    return result


def add_range_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add same-day log range/return and prior-close overnight gap features."""

    result = prices.copy()
    result["log_high_low"] = _positive_log(result["high"]) - _positive_log(result["low"])
    result["log_close_open"] = _positive_log(result["close"]) - _positive_log(result["open"])
    previous_close = result.groupby("ticker", sort=False, observed=True)["close"].shift(1)
    result["overnight_gap"] = _positive_log(result["open"]) - _positive_log(previous_close)
    return result


def add_drawdown_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add expanding-peak drawdown and trailing worst drawdowns within ticker."""

    result = prices.copy()
    running_peak = result.groupby("ticker", sort=False, observed=True)["adjusted_close"].cummax()
    result["drawdown"] = result["adjusted_close"] / running_peak - 1.0
    result["max_drawdown_20d"] = _rolling_by_ticker(result, "drawdown", 20, "min")
    result["max_drawdown_60d"] = _rolling_by_ticker(result, "drawdown", 60, "min")
    return result


def add_volume_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add current-inclusive volume ratios and log-volume-change volatility.

    Both trailing means include date t, matching the project's established
    causal rolling convention. Non-positive volume is treated as unavailable
    before taking logs; it is never replaced with zero.
    """

    result = prices.copy()
    for window in (5, 20):
        trailing_mean = _rolling_by_ticker(result, "volume", window, "mean")
        result[f"volume_ratio_{window}d"] = result["volume"] / trailing_mean.where(
            trailing_mean > 0
        )

    log_volume = _positive_log(result["volume"])
    result["_log_volume_change"] = log_volume.groupby(
        result["ticker"], sort=False, observed=True
    ).diff()
    result["volume_volatility_20d"] = _rolling_by_ticker(
        result,
        "_log_volume_change",
        20,
        "std",
    )
    return result.drop(columns="_log_volume_change")


def add_distribution_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add trailing return skew and pandas excess kurtosis within ticker."""

    result = prices.copy()
    result["rolling_skew_20d"] = _rolling_by_ticker(result, "log_return", 20, "skew")
    result["rolling_kurtosis_20d"] = _rolling_by_ticker(result, "log_return", 20, "kurt")
    return result


def create_future_rv_target(prices: pd.DataFrame) -> pd.DataFrame:
    """Add annualized RV over exactly observations t+1 through t+20 per ticker."""

    result = prices.copy()

    def future_twenty_day_mean(squared_returns: pd.Series) -> pd.Series:
        future_values = pd.concat(
            [squared_returns.shift(-offset) for offset in range(1, 21)],
            axis=1,
        )
        complete_window = future_values.notna().sum(axis=1).eq(20)
        return future_values.mean(axis=1).where(complete_window)

    future_mean = result.groupby("ticker", sort=False, observed=True)[
        "log_return_squared"
    ].transform(future_twenty_day_mean)
    result[TARGET_COLUMN] = np.sqrt(TRADING_DAYS_PER_YEAR * future_mean)
    return result
