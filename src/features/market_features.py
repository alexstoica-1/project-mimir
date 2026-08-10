"""SPY benchmark and VIX context feature calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.financial_features import TRADING_DAYS_PER_YEAR

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
SPY_RETURN_COLUMN = "_spy_log_return"
SPY_FEATURE_COLUMNS = ["market_return_20d", "market_rv_20d"]
VIX_FEATURE_COLUMNS = ["vix", "vix_change_5d"]
MARKET_CORRELATION_COLUMNS = ["market_corr_60d"]
CONTEXT_FEATURE_COLUMNS = (
    SPY_FEATURE_COLUMNS
    + MARKET_CORRELATION_COLUMNS
    + VIX_FEATURE_COLUMNS
    + ["vix_minus_market_rv"]
)


def _empty_context(columns: list[str]) -> pd.DataFrame:
    """Return an empty date-keyed context frame with a stable schema."""

    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            **{column: pd.Series(dtype=float) for column in columns},
        }
    )


def build_spy_features(spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Build date-keyed SPY log return, 20-day return, and annualized RV."""

    if spy_prices.empty:
        return _empty_context([SPY_RETURN_COLUMN, *SPY_FEATURE_COLUMNS])

    spy = spy_prices.sort_values("date", kind="stable").copy()
    log_adjusted_close = np.log(spy["adjusted_close"].where(spy["adjusted_close"] > 0))
    spy[SPY_RETURN_COLUMN] = log_adjusted_close.diff()
    spy["market_return_20d"] = log_adjusted_close.diff(periods=20)
    mean_squared_return = (
        spy[SPY_RETURN_COLUMN]
        .pow(2)
        .rolling(window=20, min_periods=20)
        .mean()
    )
    spy["market_rv_20d"] = np.sqrt(TRADING_DAYS_PER_YEAR * mean_squared_return)
    return spy.loc[:, ["date", SPY_RETURN_COLUMN, *SPY_FEATURE_COLUMNS]]


def build_vix_features(vix_prices: pd.DataFrame) -> pd.DataFrame:
    """Build decimal VIX level and five-observation percentage change by date."""

    if vix_prices.empty:
        return _empty_context(VIX_FEATURE_COLUMNS)

    vix = vix_prices.sort_values("date", kind="stable").copy()
    positive_close = vix["close"].where(vix["close"] > 0)
    vix["vix"] = positive_close / 100.0
    vix["vix_change_5d"] = positive_close.pct_change(periods=5, fill_method=None)
    return vix.loc[:, ["date", *VIX_FEATURE_COLUMNS]]


def add_market_correlation(stocks: pd.DataFrame, spy_features: pd.DataFrame) -> pd.DataFrame:
    """Add 60-day stock/SPY return correlation after exact-date alignment."""

    result = stocks.merge(
        spy_features.loc[:, ["date", SPY_RETURN_COLUMN]],
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    correlation = pd.Series(np.nan, index=result.index, dtype=float)
    for _, ticker_rows in result.groupby("ticker", sort=False, observed=True):
        correlation.loc[ticker_rows.index] = (
            ticker_rows["log_return"]
            .rolling(window=60, min_periods=60)
            .corr(ticker_rows[SPY_RETURN_COLUMN])
            .to_numpy()
        )
    result["market_corr_60d"] = correlation
    return result.drop(columns=SPY_RETURN_COLUMN)


def merge_context_features(
    stocks: pd.DataFrame,
    spy_features: pd.DataFrame,
    vix_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge exact-date SPY and VIX context onto prediction-stock rows."""

    result = stocks.merge(
        spy_features.loc[:, ["date", *SPY_FEATURE_COLUMNS]],
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.merge(
        vix_features,
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["vix_minus_market_rv"] = result["vix"] - result["market_rv_20d"]
    return result
