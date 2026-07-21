from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd

from scripts import ingest_market_data
from src.services import MarketDataIngestionError, MarketDataIngestionService
from src.validation import ValidationResult


@dataclass
class Run:
    id: int


def price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "adjusted_close": [103.5, 104.5],
            "volume": [1000, 1200],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 0.0],
            "source": ["yfinance", "yfinance"],
        }
    )


def metadata() -> dict[str, object]:
    return {"ticker": "AAPL", "company_name": "Apple Inc.", "source": "yfinance"}


def valid_result(row_count: int = 1, warnings: list[str] | None = None) -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings or [],
        row_count=row_count,
    )


def invalid_result(error: str) -> ValidationResult:
    return ValidationResult(is_valid=False, errors=[error], warnings=[], row_count=1)


def dependencies():
    collector = Mock()
    validator = Mock()
    repository = Mock()
    repository.create_ingestion_run.return_value = Run(id=123)
    repository.latest_price_date.return_value = None
    collector.fetch_company_metadata.return_value = metadata()
    collector.fetch_daily_prices.return_value = price_frame()
    validator.validate_company_metadata.return_value = valid_result()
    validator.validate_daily_prices.return_value = valid_result(row_count=2)
    repository.upsert_daily_prices.return_value = 2
    return collector, validator, repository


def test_successful_first_ingestion() -> None:
    collector, validator, repository = dependencies()

    summary = MarketDataIngestionService(collector, validator, repository).ingest_ticker("aapl")

    assert summary.status == "completed"
    assert summary.ticker == "AAPL"
    assert summary.rows_fetched == 2
    assert summary.rows_written == 2
    assert summary.earliest_date == date(2024, 1, 2)
    assert summary.latest_date == date(2024, 1, 3)
    repository.complete_ingestion_run.assert_called_once_with(
        123,
        rows_fetched=2,
        rows_written=2,
        warning_count=0,
        error_count=0,
    )


def test_successful_incremental_ingestion() -> None:
    collector, validator, repository = dependencies()
    repository.latest_price_date.return_value = date(2024, 1, 3)

    summary = MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    assert summary.status == "completed"
    collector.fetch_daily_prices.assert_called_once_with(
        "AAPL",
        start="2024-01-04",
        end=None,
        period=None,
    )


def test_company_upsert_is_called() -> None:
    collector, validator, repository = dependencies()

    MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    repository.upsert_company.assert_called_once_with(metadata(), commit=False)


def test_price_upsert_is_called() -> None:
    collector, validator, repository = dependencies()

    MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    repository.upsert_daily_prices.assert_called_once()
    assert repository.upsert_daily_prices.call_args.kwargs == {"commit": False}


def test_latest_stored_date_is_used() -> None:
    collector, validator, repository = dependencies()

    MarketDataIngestionService(collector, validator, repository).ingest_ticker("msft")

    repository.latest_price_date.assert_called_once_with("MSFT", source="yfinance")


def test_one_day_is_added_after_latest_stored_date() -> None:
    collector, validator, repository = dependencies()
    repository.latest_price_date.return_value = date(2024, 5, 10)

    MarketDataIngestionService(collector, validator, repository).ingest_ticker("MSFT")

    assert collector.fetch_daily_prices.call_args.kwargs["start"] == "2024-05-11"


def test_no_new_data_returns_successful_zero_row_result() -> None:
    collector, validator, repository = dependencies()
    repository.latest_price_date.return_value = date(2024, 1, 3)
    collector.fetch_daily_prices.return_value = pd.DataFrame()

    summary = MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    assert summary.status == "completed"
    assert summary.rows_fetched == 0
    assert summary.rows_written == 0
    validator.validate_daily_prices.assert_not_called()
    repository.upsert_daily_prices.assert_not_called()


def test_invalid_metadata_blocks_persistence() -> None:
    collector, validator, repository = dependencies()
    validator.validate_company_metadata.return_value = invalid_result("bad metadata")

    summary = MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    assert summary.status == "failed"
    repository.upsert_company.assert_not_called()
    repository.upsert_daily_prices.assert_not_called()
    repository.fail_ingestion_run.assert_called_once()


def test_invalid_prices_blocks_persistence() -> None:
    collector, validator, repository = dependencies()
    validator.validate_daily_prices.return_value = invalid_result("bad prices")

    summary = MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    assert summary.status == "failed"
    repository.upsert_company.assert_not_called()
    repository.upsert_daily_prices.assert_not_called()
    repository.fail_ingestion_run.assert_called_once()


def test_collector_failure_marks_run_failed() -> None:
    collector, validator, repository = dependencies()
    collector.fetch_company_metadata.side_effect = RuntimeError("network failed")

    try:
        MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")
    except MarketDataIngestionError:
        pass

    repository.fail_ingestion_run.assert_called_once()


def test_repository_failure_marks_run_failed() -> None:
    collector, validator, repository = dependencies()
    repository.upsert_daily_prices.side_effect = RuntimeError("database failed")

    try:
        MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")
    except MarketDataIngestionError:
        pass

    repository.rollback.assert_called()
    repository.fail_ingestion_run.assert_called_once()


def test_completed_run_records_correct_counts() -> None:
    collector, validator, repository = dependencies()
    validator.validate_company_metadata.return_value = valid_result(warnings=["metadata warning"])
    validator.validate_daily_prices.return_value = valid_result(row_count=2, warnings=["price warning"])

    MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")

    repository.complete_ingestion_run.assert_called_once_with(
        123,
        rows_fetched=2,
        rows_written=2,
        warning_count=2,
        error_count=0,
    )


def test_multiple_ticker_cli_behavior() -> None:
    summaries = [
        ingest_market_data.IngestionSummary(
            ticker="AAPL",
            status="completed",
            rows_fetched=2,
            rows_written=2,
            warning_count=0,
            error_count=0,
            earliest_date=date(2024, 1, 2),
            latest_date=date(2024, 1, 3),
            ingestion_run_id=1,
            message="ok",
        ),
        ingest_market_data.IngestionSummary(
            ticker="MSFT",
            status="completed",
            rows_fetched=1,
            rows_written=1,
            warning_count=0,
            error_count=0,
            earliest_date=date(2024, 1, 2),
            latest_date=date(2024, 1, 2),
            ingestion_run_id=2,
            message="ok",
        ),
    ]

    with patch("scripts.ingest_market_data.ingest_one", side_effect=summaries) as ingest_one:
        exit_code = ingest_market_data.main_args(["--tickers", "AAPL", "MSFT", "--period", "1y"])

    assert exit_code == 0
    assert ingest_one.call_count == 2


def test_input_ticker_normalization() -> None:
    collector, validator, repository = dependencies()

    MarketDataIngestionService(collector, validator, repository).ingest_ticker(" nvda ")

    repository.create_ingestion_run.assert_called_once_with(
        source="yfinance",
        ticker="NVDA",
        status="running",
    )


def test_dependency_injection() -> None:
    collector, validator, repository = dependencies()
    service = MarketDataIngestionService(collector, validator, repository)

    assert service.collector is collector
    assert service.validator is validator
    assert service.repository is repository


def test_no_partial_write_behavior() -> None:
    collector, validator, repository = dependencies()
    repository.upsert_daily_prices.side_effect = RuntimeError("price write failed")

    try:
        MarketDataIngestionService(collector, validator, repository).ingest_ticker("AAPL")
    except MarketDataIngestionError:
        pass

    repository.upsert_company.assert_called_once_with(metadata(), commit=False)
    repository.commit.assert_not_called()
    repository.rollback.assert_called()
