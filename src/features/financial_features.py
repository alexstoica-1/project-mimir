"""Causal financial feature calculations for normalized daily prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_FEATURE_COLUMNS = ["return_1d", "return_5d", "return_10d", "return_20d"]
VOLATILITY_FEATURE_COLUMNS = [
    "volatility_5d",
    "volatility_20d",
    "volatility_ratio_5d_20d",
]
DRAWDOWN_FEATURE_COLUMNS = ["drawdown_20d"]


def add_return_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add close-to-close returns using only the current and prior closes.

    ``prices`` must already be ordered by date ascending. The input frame is
    never modified.
    """

    result = prices.copy()
    close = result["close"]
    for window in (1, 5, 10, 20):
        result[f"return_{window}d"] = close.pct_change(periods=window, fill_method=None)
    return result


def add_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add trailing volatility features from the already-computed 1-day return.

    Each rolling window includes its current row and only earlier rows. The
    sample standard deviation is used, which is the pandas default.
    """

    result = prices.copy()
    daily_return = result["return_1d"]
    result["volatility_5d"] = daily_return.rolling(window=5, min_periods=5).std()
    result["volatility_20d"] = daily_return.rolling(window=20, min_periods=20).std()
    result["volatility_ratio_5d_20d"] = result["volatility_5d"] / result["volatility_20d"]
    result["volatility_ratio_5d_20d"] = result["volatility_ratio_5d_20d"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return result


def add_drawdown_feature(prices: pd.DataFrame) -> pd.DataFrame:
    """Add the current close's drawdown from its trailing 20-day peak."""

    result = prices.copy()
    rolling_peak = result["close"].rolling(window=20, min_periods=20).max()
    result["drawdown_20d"] = result["close"] / rolling_peak - 1.0
    return result
