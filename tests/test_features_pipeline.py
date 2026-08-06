"""Tests for the database-backed market feature pipeline."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pandas.testing as pdt

from src.features.pipeline import FEATURE_COLUMNS, OUTPUT_COLUMNS, MarketFeaturePipeline


def synthetic_prices(ticker: str = "AAPL", periods: int = 30) -> pd.DataFrame:
    """Create a deterministic, non-constant normalized daily-price frame."""

    close = pd.Series([100.0 + index * 1.5 + (index % 3) * 0.2 for index in range(periods)])
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2024-01-02", periods=periods),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1_000 + index for index in range(periods)],
            "adjusted_close": close - 0.1,
            "source": "yfinance",
        }
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
        frame = self.frames[ticker].copy()
        if start is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
        return frame


def test_computes_documented_features_and_drops_lookback_rows() -> None:
    features = MarketFeaturePipeline.transform_price_history(synthetic_prices())

    assert list(features.columns) == OUTPUT_COLUMNS
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == 10
    assert features[FEATURE_COLUMNS].notna().all().all()


def test_transform_does_not_mutate_source_price_history() -> None:
    prices = synthetic_prices().sample(frac=1, random_state=7).reset_index(drop=True)
    original = prices.copy(deep=True)

    MarketFeaturePipeline.transform_price_history(prices)

    pdt.assert_frame_equal(prices, original)


def test_feature_rows_are_sorted_and_do_not_leak_future_prices() -> None:
    prices = synthetic_prices()
    baseline = MarketFeaturePipeline.transform_price_history(prices)
    changed_future = prices.copy()
    changed_future.loc[25, "close"] *= 5

    changed = MarketFeaturePipeline.transform_price_history(changed_future)
    comparison_end = prices.loc[24, "date"]
    baseline_before_change = baseline.loc[baseline["date"] <= comparison_end, FEATURE_COLUMNS]
    changed_before_change = changed.loc[changed["date"] <= comparison_end, FEATURE_COLUMNS]

    assert features_are_sorted(baseline)
    pdt.assert_frame_equal(
        baseline_before_change.reset_index(drop=True),
        changed_before_change.reset_index(drop=True),
    )


def test_builds_one_ticker_through_the_repository_contract() -> None:
    repository = FakeRepository({"AAPL": synthetic_prices("AAPL")})
    pipeline = MarketFeaturePipeline(repository)

    features = pipeline.build_for_ticker("AAPL")

    assert repository.requested_tickers == ["AAPL"]
    assert features["ticker"].eq("AAPL").all()


def test_builds_and_combines_each_ticker_independently() -> None:
    repository = FakeRepository(
        {
            "MSFT": synthetic_prices("MSFT"),
            "AAPL": synthetic_prices("AAPL"),
        }
    )

    features = MarketFeaturePipeline(repository).build_for_universe()

    assert repository.requested_tickers == ["AAPL", "MSFT"]
    assert features["ticker"].unique().tolist() == ["AAPL", "MSFT"]
    assert features.groupby("ticker")["date"].apply(lambda dates: dates.is_monotonic_increasing).all()


def test_saves_model_ready_csv_to_processed_path(tmp_path) -> None:
    features = MarketFeaturePipeline.transform_price_history(synthetic_prices())
    output_path = tmp_path / "data" / "processed" / "v1" / "features_all_tickers.csv"

    saved_path = MarketFeaturePipeline.save_csv(features, output_path)
    saved = pd.read_csv(saved_path)

    assert saved_path == output_path
    assert list(saved.columns) == OUTPUT_COLUMNS
    assert len(saved) == len(features)


def features_are_sorted(features: pd.DataFrame) -> bool:
    """Keep date-order assertion readable in leakage-focused tests."""

    return features["date"].is_monotonic_increasing
