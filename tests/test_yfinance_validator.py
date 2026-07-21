from __future__ import annotations

import pandas as pd

from src.validation import YFinanceDataValidator


def valid_daily_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "adjusted_close": [103.5, 104.5, 105.5],
            "volume": [1000, 1200, 1300],
            "dividends": [0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0],
            "source": ["yfinance", "yfinance", "yfinance"],
        }
    )


def valid_metadata() -> dict[str, object]:
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NMS",
        "currency": "USD",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 1_000_000,
        "source": "yfinance",
    }


def error_text(result) -> str:
    return " ".join(result.errors)


def warning_text(result) -> str:
    return " ".join(result.warnings)


def test_valid_data() -> None:
    validator = YFinanceDataValidator()

    price_result = validator.validate_daily_prices(valid_daily_prices())
    metadata_result = validator.validate_company_metadata(valid_metadata())

    assert price_result.is_valid
    assert price_result.errors == []
    assert price_result.warnings == []
    assert price_result.row_count == 3
    assert metadata_result.is_valid
    assert metadata_result.errors == []


def test_missing_required_columns() -> None:
    df = valid_daily_prices().drop(columns=["close"])

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "Missing required columns" in error_text(result)


def test_duplicate_ticker_date() -> None:
    df = pd.concat([valid_daily_prices(), valid_daily_prices().iloc[[0]]], ignore_index=True)

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "Duplicate ticker/date pairs" in error_text(result)
    assert "Duplicate full rows" in warning_text(result)


def test_invalid_ohlc_relationships() -> None:
    df = valid_daily_prices()
    df.loc[0, "high"] = 98.0
    df.loc[1, "low"] = 108.0

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "high is less than low" in error_text(result)
    assert "high is less than open" in error_text(result)
    assert "high is less than close" in error_text(result)
    assert "low is greater than open" in error_text(result)
    assert "low is greater than close" in error_text(result)


def test_negative_prices() -> None:
    df = valid_daily_prices()
    df.loc[0, "open"] = -1.0

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "negative prices" in error_text(result)


def test_negative_volume() -> None:
    df = valid_daily_prices()
    df.loc[0, "volume"] = -100

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "negative volume" in error_text(result)


def test_future_dates() -> None:
    df = valid_daily_prices()
    df.loc[0, "date"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert not result.is_valid
    assert "future" in error_text(result)


def test_missing_optional_metadata() -> None:
    metadata = valid_metadata()
    metadata["sector"] = None
    metadata["industry"] = None

    result = YFinanceDataValidator().validate_company_metadata(metadata)

    assert result.is_valid
    assert "sector is missing" in warning_text(result)
    assert "industry is missing" in warning_text(result)


def test_invalid_source() -> None:
    df = valid_daily_prices()
    df.loc[0, "source"] = "manual"
    metadata = valid_metadata()
    metadata["source"] = "manual"

    price_result = YFinanceDataValidator().validate_daily_prices(df)
    metadata_result = YFinanceDataValidator().validate_company_metadata(metadata)

    assert not price_result.is_valid
    assert not metadata_result.is_valid
    assert "source is missing" in error_text(price_result)
    assert "source is missing" in error_text(metadata_result)


def test_unsorted_dates_warning() -> None:
    df = valid_daily_prices().iloc[[1, 0, 2]].reset_index(drop=True)

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert result.is_valid
    assert "dates are not sorted ascending" in warning_text(result)


def test_large_daily_movement_warning() -> None:
    df = valid_daily_prices()
    df.loc[1, "close"] = 250.0
    df.loc[1, "high"] = 260.0

    result = YFinanceDataValidator().validate_daily_prices(df)

    assert result.is_valid
    assert "unusually large one-day close-to-close movement" in warning_text(result)


def test_original_dataframe_is_not_mutated() -> None:
    df = valid_daily_prices()
    original = df.copy(deep=True)

    YFinanceDataValidator().validate_daily_prices(df)

    pd.testing.assert_frame_equal(df, original)
