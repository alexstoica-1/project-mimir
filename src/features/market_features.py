"""Causal intraday market features for normalized daily prices."""

from __future__ import annotations

import pandas as pd

MARKET_FEATURE_COLUMNS = ["high_low_range_pct", "close_open_return"]


def add_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add same-session range and close-to-open features without mutation."""

    result = prices.copy()
    result["high_low_range_pct"] = (result["high"] - result["low"]) / result["close"]
    result["close_open_return"] = (result["close"] - result["open"]) / result["open"]
    return result
