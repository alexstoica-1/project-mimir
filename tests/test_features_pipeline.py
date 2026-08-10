"""Deterministic tests for the database-backed market feature pipeline."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.financial_features import (
    TRADING_DAYS_PER_YEAR,
    add_realized_volatility_features,
    add_return_features,
    create_future_rv_target,
)
from src.features.market_features import (
    SPY_RETURN_COLUMN,
    add_market_correlation,
    build_spy_features,
)
from src.features.pipeline import (
    BASE_PRICE_COLUMNS,
    FEATURE_COLUMNS,
    INFERENCE_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    MarketFeaturePipeline,
    validate_and_sort_raw_data,
)


def synthetic_prices(
    ticker: str = "AAPL",
    periods: int = 130,
    *,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Create deterministic, non-constant normalized daily prices."""

    row_dates = dates if dates is not None else pd.bdate_range("2024-01-02", periods=periods)
    steps = np.arange(periods, dtype=float)
    if ticker == "^VIX":
        close = 20.0 + 0.03 * steps + 0.4 * np.sin(steps / 4.0)
    else:
        ticker_offset = 0.0001 if ticker == "MSFT" else 0.0
        log_price = np.log(100.0) + (0.001 + ticker_offset) * steps + 0.002 * np.sin(steps)
        close = np.exp(log_price)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": row_dates,
            "open": close * 0.997,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000 + (steps * 1_003).astype(int),
            "adjusted_close": close,
            "source": "yfinance",
        }
    )


def synthetic_market_data(*tickers: str, periods: int = 130) -> pd.DataFrame:
    """Combine prediction tickers with same-calendar SPY and VIX context."""

    prediction_tickers = tickers or ("AAPL",)
    return pd.concat(
        [
            *(synthetic_prices(ticker, periods) for ticker in prediction_tickers),
            synthetic_prices("SPY", periods),
            synthetic_prices("^VIX", periods),
        ],
        ignore_index=True,
    )


class FakeRepository:
    """Small in-memory stand-in for the repository's feature read contract."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.requested_tickers: list[str] = []

    def list_available_companies(self, source: str = "yfinance") -> list[str]:
        return sorted(self.frames)

    def get_price_history_frame(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        self.requested_tickers.append(ticker)
        frame = self.frames.get(ticker, pd.DataFrame(columns=BASE_PRICE_COLUMNS)).copy()
        if start is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
        return frame


def test_log_returns_are_independent_within_ticker_and_never_cross_boundaries() -> None:
    aapl = synthetic_prices("AAPL", 3)
    msft = synthetic_prices("MSFT", 3)
    msft["adjusted_close"] = [200.0, 220.0, 242.0]
    combined = pd.concat([msft.iloc[::-1], aapl.iloc[::-1]], ignore_index=True)

    sorted_prices = validate_and_sort_raw_data(combined)
    features = add_return_features(sorted_prices)
    first_rows = features.groupby("ticker", sort=False).head(1)

    assert first_rows["log_return"].isna().all()
    msft_returns = features.loc[features["ticker"].eq("MSFT"), "log_return"]
    assert msft_returns.iloc[1] == pytest.approx(np.log(220.0 / 200.0))


def test_realized_volatility_uses_squared_log_returns_and_252_annualization() -> None:
    returns = np.array([0.0, 0.01, -0.02, 0.03, -0.04, 0.05])
    frame = pd.DataFrame(
        {
            "ticker": "AAPL",
            "log_return": returns,
            "log_return_squared": returns**2,
        }
    )

    features = add_realized_volatility_features(frame)
    expected = np.sqrt(TRADING_DAYS_PER_YEAR * np.mean(returns[1:6] ** 2))

    assert features.loc[5, "rv_5d"] == pytest.approx(expected)


def test_realized_volatility_lags_are_ticker_specific() -> None:
    rows = []
    for ticker, level in (("AAPL", 0.01), ("MSFT", 0.03)):
        rows.extend(
            {
                "ticker": ticker,
                "log_return": level,
                "log_return_squared": level**2,
            }
            for _ in range(45)
        )
    features = add_realized_volatility_features(pd.DataFrame(rows))
    msft = features.loc[features["ticker"].eq("MSFT")].reset_index(drop=True)

    assert pd.isna(msft.loc[0, "rv_20d_lag1"])
    assert msft.loc[20, "rv_20d_lag1"] == pytest.approx(np.sqrt(252 * 0.03**2))
    assert msft.loc[39, "rv_20d_lag20"] == pytest.approx(np.sqrt(252 * 0.03**2))


def test_future_target_uses_exactly_t_plus_1_through_t_plus_20() -> None:
    squared_returns = pd.Series(np.arange(22, dtype=float) ** 2)
    frame = pd.DataFrame(
        {
            "ticker": "AAPL",
            "log_return_squared": squared_returns,
        }
    )

    targeted = create_future_rv_target(frame)
    expected_t0 = np.sqrt(252 * np.mean(np.arange(1, 21, dtype=float) ** 2))
    expected_t1 = np.sqrt(252 * np.mean(np.arange(2, 22, dtype=float) ** 2))

    assert targeted.loc[0, "target_rv_20d"] == pytest.approx(expected_t0)
    assert targeted.loc[1, "target_rv_20d"] == pytest.approx(expected_t1)
    assert targeted.loc[2:, "target_rv_20d"].isna().all()


def test_spy_features_merge_by_actual_date() -> None:
    raw = synthetic_market_data("AAPL", periods=80)
    intermediate = MarketFeaturePipeline.transform_price_history(raw, model_ready=False)
    spy = synthetic_prices("SPY", 80)
    expected = np.log(spy.loc[20, "adjusted_close"] / spy.loc[0, "adjusted_close"]) # type: ignore
    row = intermediate.loc[intermediate["date"].eq(spy.loc[20, "date"])].iloc[0]

    assert row["market_return_20d"] == pytest.approx(expected)


def test_stock_spy_correlation_requires_exact_date_alignment() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    signal = np.sin(np.arange(100, dtype=float) / 5.0)
    stocks = pd.DataFrame({"ticker": "AAPL", "date": dates, "log_return": signal})
    spy_features = pd.DataFrame(
        {
            "date": dates.delete(30),
            SPY_RETURN_COLUMN: np.delete(signal, 30),
        }
    )

    correlated = add_market_correlation(stocks, spy_features)

    assert pd.isna(correlated.loc[65, "market_corr_60d"])
    assert correlated.loc[99, "market_corr_60d"] == pytest.approx(1.0)


def test_vix_is_decimal_before_subtracting_market_realized_volatility() -> None:
    raw = synthetic_market_data("AAPL", periods=80)
    intermediate = MarketFeaturePipeline.transform_price_history(raw, model_ready=False)
    row = intermediate.iloc[30]
    vix_close = synthetic_prices("^VIX", 80).loc[30, "close"]

    assert row["vix"] == pytest.approx(vix_close / 100.0) # type: ignore 
    assert row["vix_minus_market_rv"] == pytest.approx(row["vix"] - row["market_rv_20d"])


def test_missing_context_is_preserved_as_nan_until_final_filter() -> None:
    intermediate = MarketFeaturePipeline.transform_price_history(
        synthetic_prices("AAPL", 80),
        model_ready=False,
    )
    model_ready = MarketFeaturePipeline.transform_price_history(synthetic_prices("AAPL", 80))

    assert intermediate[["market_rv_20d", "vix"]].isna().all().all()
    assert model_ready.empty


def test_context_assets_are_excluded_and_final_data_are_deterministically_sorted() -> None:
    raw = synthetic_market_data("MSFT", "AAPL").sample(frac=1, random_state=7)
    features = MarketFeaturePipeline.transform_price_history(raw)

    assert not features["ticker"].isin(["SPY", "^VIX"]).any()
    assert features["ticker"].unique().tolist() == ["AAPL", "MSFT"]
    assert features.groupby("ticker")["date"].apply(lambda values: values.is_monotonic_increasing).all()


def test_no_input_feature_uses_prices_later_than_its_date() -> None:
    raw = synthetic_market_data("AAPL")
    baseline = MarketFeaturePipeline.transform_price_history(raw, model_ready=False)
    changed_future = raw.copy()
    future_mask = changed_future["ticker"].eq("AAPL") & changed_future["date"].eq(
        pd.bdate_range("2024-01-02", periods=130)[90]
    )
    changed_future.loc[future_mask, "adjusted_close"] *= 5  # type: ignore[index]
    changed_future.loc[future_mask, ["open", "high", "low", "close", "volume"]] *= 5

    changed = MarketFeaturePipeline.transform_price_history(changed_future, model_ready=False)
    comparison_end = pd.bdate_range("2024-01-02", periods=130)[89]
    baseline_past = baseline.loc[baseline["date"] <= comparison_end, FEATURE_COLUMNS]
    changed_past = changed.loc[changed["date"] <= comparison_end, FEATURE_COLUMNS]

    pdt.assert_frame_equal(
        baseline_past.reset_index(drop=True),
        changed_past.reset_index(drop=True),
    )


def test_model_ready_output_has_documented_columns_and_no_missing_features() -> None:
    features = MarketFeaturePipeline.transform_price_history(synthetic_market_data("AAPL"))

    assert list(features.columns) == OUTPUT_COLUMNS
    assert len(features) > 0
    assert features[[*FEATURE_COLUMNS, "target_rv_20d"]].notna().all().all()


def test_inference_output_keeps_latest_complete_feature_row_without_target() -> None:
    raw = synthetic_market_data("AAPL", periods=130)

    features = MarketFeaturePipeline.transform_price_history(raw, inference_ready=True)

    assert list(features.columns) == INFERENCE_OUTPUT_COLUMNS
    assert "target_rv_20d" not in features.columns
    assert features[FEATURE_COLUMNS].notna().all().all()
    assert features["date"].max() == raw.loc[raw["ticker"].eq("AAPL"), "date"].max()


def test_transform_does_not_mutate_source_price_history() -> None:
    prices = synthetic_market_data("AAPL").sample(frac=1, random_state=7).reset_index(drop=True)
    original = prices.copy(deep=True)

    MarketFeaturePipeline.transform_price_history(prices)

    pdt.assert_frame_equal(prices, original)


def test_builds_one_ticker_and_loads_required_context_through_repository() -> None:
    repository = FakeRepository(
        {
            "AAPL": synthetic_prices("AAPL"),
            "SPY": synthetic_prices("SPY"),
            "^VIX": synthetic_prices("^VIX"),
        }
    )

    features = MarketFeaturePipeline(repository).build_for_ticker("AAPL")

    assert repository.requested_tickers == ["AAPL", "SPY", "^VIX"]
    assert features["ticker"].eq("AAPL").all()


def test_builds_universe_without_emitting_context_tickers() -> None:
    repository = FakeRepository(
        {
            "MSFT": synthetic_prices("MSFT"),
            "AAPL": synthetic_prices("AAPL"),
            "SPY": synthetic_prices("SPY"),
            "^VIX": synthetic_prices("^VIX"),
        }
    )

    features = MarketFeaturePipeline(repository).build_for_universe()

    assert repository.requested_tickers == ["AAPL", "MSFT", "SPY", "^VIX"]
    assert features["ticker"].unique().tolist() == ["AAPL", "MSFT"]


def test_saves_model_ready_csv_to_processed_path(tmp_path) -> None:
    features = MarketFeaturePipeline.transform_price_history(synthetic_market_data("AAPL"))
    output_path = tmp_path / "data" / "processed" / "v1" / "features_all_tickers.csv"

    saved_path = MarketFeaturePipeline.save_csv(features, output_path)
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert list(saved.columns) == OUTPUT_COLUMNS
    assert len(saved) == len(features)
