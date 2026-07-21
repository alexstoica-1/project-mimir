from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.collectors.yfinance_collector import YFinanceCollectionError, YFinanceCollector


def make_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Adj Close": [10.5, 11.5],
            "Volume": [100, 200],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
    )


def test_ticker_normalization_for_metadata() -> None:
    ticker_obj = Mock()
    ticker_obj.get_info.return_value = {"longName": "Apple Inc.", "currency": "USD"}
    ticker_obj.fast_info = {}
    ticker_obj.get_history_metadata.return_value = {}

    with patch("src.collectors.yfinance_collector.yf.Ticker", return_value=ticker_obj) as ticker_mock:
        metadata = YFinanceCollector().fetch_company_metadata(" aapl ")

    ticker_mock.assert_called_once_with("AAPL")
    assert metadata["ticker"] == "AAPL"


def test_invalid_empty_ticker() -> None:
    with pytest.raises(YFinanceCollectionError, match="ticker"):
        YFinanceCollector().fetch_daily_prices("   ")


def test_invalid_start_end_ordering() -> None:
    with pytest.raises(YFinanceCollectionError, match="start must be earlier"):
        YFinanceCollector().fetch_daily_prices("AAPL", start="2024-02-01", end="2024-01-01")


def test_normalized_output_column_names_and_order() -> None:
    prices = YFinanceCollector._normalize_daily_prices(make_price_frame(), "AAPL")

    assert list(prices.columns) == YFinanceCollector.PRICE_COLUMNS


def test_duplicate_removal() -> None:
    raw_prices = pd.concat([make_price_frame(), make_price_frame().iloc[[0]]])

    prices = YFinanceCollector._normalize_daily_prices(raw_prices, "AAPL")

    assert len(prices) == 2
    assert int(prices.duplicated().sum()) == 0


def test_timezone_removal_from_date() -> None:
    raw_prices = make_price_frame()
    raw_prices.index = pd.DatetimeIndex(
        ["2024-01-02 09:30:00-05:00", "2024-01-03 09:30:00-05:00"],
        name="Date",
    )

    prices = YFinanceCollector._normalize_daily_prices(raw_prices, "AAPL")

    assert prices["date"].dt.tz is None


def test_missing_optional_metadata_handling() -> None:
    ticker_obj = Mock()
    ticker_obj.get_info.return_value = {"longName": "Apple Inc."}
    ticker_obj.fast_info = {}
    ticker_obj.get_history_metadata.return_value = {}

    with patch("src.collectors.yfinance_collector.yf.Ticker", return_value=ticker_obj):
        metadata = YFinanceCollector().fetch_company_metadata("AAPL")

    assert metadata["company_name"] == "Apple Inc."
    assert metadata["sector"] is None
    assert metadata["source"] == "yfinance"


def test_empty_api_response_returns_empty_schema() -> None:
    raw_prices = pd.DataFrame()

    prices = YFinanceCollector._normalize_daily_prices(raw_prices, "AAPL")

    assert prices.empty
    assert list(prices.columns) == YFinanceCollector.PRICE_COLUMNS


def test_yfinance_exception_handling() -> None:
    with patch("src.collectors.yfinance_collector.yf.download", side_effect=RuntimeError("boom")):
        with pytest.raises(YFinanceCollectionError, match="Failed to fetch"):
            YFinanceCollector().fetch_daily_prices("AAPL")


def test_multiindex_column_handling() -> None:
    raw_prices = make_price_frame()
    raw_prices.columns = pd.MultiIndex.from_product([raw_prices.columns, ["AAPL"]])

    prices = YFinanceCollector._normalize_daily_prices(raw_prices, "AAPL")

    assert list(prices.columns) == YFinanceCollector.PRICE_COLUMNS
    assert prices["ticker"].unique().tolist() == ["AAPL"]


def test_fetch_daily_prices_passes_date_range_without_period() -> None:
    with patch("src.collectors.yfinance_collector.yf.download", return_value=make_price_frame()) as download:
        prices = YFinanceCollector().fetch_daily_prices(
            "msft",
            start="2024-01-01",
            end="2024-02-01",
        )

    kwargs = download.call_args.kwargs
    assert kwargs["tickers"] == "MSFT"
    assert kwargs["start"] == "2024-01-01"
    assert kwargs["end"] == "2024-02-01"
    assert "period" not in kwargs
    assert prices["ticker"].unique().tolist() == ["MSFT"]
