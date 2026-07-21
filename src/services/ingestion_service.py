"""End-to-end market data ingestion service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.validation import ValidationResult

logger = logging.getLogger(__name__)


class MarketDataIngestionError(RuntimeError):
    """Raised when an unexpected ingestion failure occurs."""


@dataclass(frozen=True)
class IngestionSummary:
    """Summary of one ticker ingestion attempt."""

    ticker: str
    status: str
    rows_fetched: int
    rows_written: int
    warning_count: int
    error_count: int
    earliest_date: date | None
    latest_date: date | None
    ingestion_run_id: int | None
    message: str


@dataclass(frozen=True)
class _PriceRequest:
    start: str | None
    end: str | None
    period: str | None
    should_fetch: bool


class MarketDataIngestionService:
    """Orchestrate collection, validation, and repository persistence."""

    def __init__(self, collector: Any, validator: Any, repository: Any) -> None:
        """Create a service with explicit injected dependencies."""

        self.collector = collector
        self.validator = validator
        self.repository = repository

    def ingest_ticker(
        self,
        ticker: str,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
        incremental: bool = True,
    ) -> IngestionSummary:
        """Ingest one ticker from yfinance into the repository."""

        normalized_ticker = self._normalize_ticker(ticker)
        ingestion_run_id: int | None = None
        rows_fetched = 0
        rows_written = 0
        warning_count = 0
        error_count = 0
        logger.info(
            "Starting market data ingestion ticker=%s period=%s start=%s end=%s incremental=%s",
            normalized_ticker,
            period,
            start,
            end,
            incremental,
        )

        try:
            run = self.repository.create_ingestion_run(
                source="yfinance",
                ticker=normalized_ticker,
                status="running",
            )
            ingestion_run_id = getattr(run, "id", None)
        except Exception as exc:
            raise MarketDataIngestionError(
                f"Failed to create ingestion run for {normalized_ticker}"
            ) from exc

        try:
            metadata = self.collector.fetch_company_metadata(normalized_ticker)
            price_request = self._build_price_request(
                ticker=normalized_ticker,
                period=period,
                start=start,
                end=end,
                incremental=incremental,
            )

            if price_request.should_fetch:
                prices = self.collector.fetch_daily_prices(
                    normalized_ticker,
                    start=price_request.start,
                    end=price_request.end,
                    period=price_request.period,
                )
            else:
                prices = self._empty_price_frame()

            rows_fetched = len(prices)
            metadata_result = self.validator.validate_company_metadata(metadata)
            prices_result = self._validate_prices(prices, incremental=incremental)
            warning_count = len(metadata_result.warnings) + len(prices_result.warnings)
            error_count = len(metadata_result.errors) + len(prices_result.errors)

            if not metadata_result.is_valid or not prices_result.is_valid:
                message = self._validation_message(metadata_result, prices_result)
                self.repository.fail_ingestion_run(
                    ingestion_run_id,
                    error_message=message,
                    rows_fetched=rows_fetched,
                    rows_written=0,
                    warning_count=warning_count,
                    error_count=error_count,
                )
                logger.warning("Market data validation failed ticker=%s errors=%s", normalized_ticker, error_count)
                return self._summary(
                    ticker=normalized_ticker,
                    status="failed",
                    rows_fetched=rows_fetched,
                    rows_written=0,
                    warning_count=warning_count,
                    error_count=error_count,
                    prices=prices,
                    ingestion_run_id=ingestion_run_id,
                    message=message,
                )

            try:
                self.repository.upsert_company(metadata, commit=False)
                if prices.empty:
                    rows_written = 0
                else:
                    rows_written = self.repository.upsert_daily_prices(prices, commit=False)
                self.repository.commit(f"ingest ticker {normalized_ticker}")
            except Exception:
                self.repository.rollback()
                raise

            self.repository.complete_ingestion_run(
                ingestion_run_id,
                rows_fetched=rows_fetched,
                rows_written=rows_written,
                warning_count=warning_count,
                error_count=0,
            )
            message = "No new price rows to ingest." if rows_fetched == 0 else "Ingestion completed."
            return self._summary(
                ticker=normalized_ticker,
                status="completed",
                rows_fetched=rows_fetched,
                rows_written=rows_written,
                warning_count=warning_count,
                error_count=0,
                prices=prices,
                ingestion_run_id=ingestion_run_id,
                message=message,
            )
        except MarketDataIngestionError:
            raise
        except Exception as exc:
            self._mark_failed_after_exception(
                ingestion_run_id=ingestion_run_id,
                rows_fetched=rows_fetched,
                rows_written=rows_written,
                warning_count=warning_count,
                error_count=max(error_count, 1),
                error_message=str(exc),
            )
            raise MarketDataIngestionError(
                f"Market data ingestion failed for {normalized_ticker}"
            ) from exc

    def _build_price_request(
        self,
        *,
        ticker: str,
        period: str,
        start: str | None,
        end: str | None,
        incremental: bool,
    ) -> _PriceRequest:
        start_date = self._parse_date(start, "start") if start is not None else None
        end_date = self._parse_date(end, "end") if end is not None else None
        if start_date is not None and end_date is not None and start_date >= end_date:
            raise MarketDataIngestionError("start must be earlier than end")

        if not incremental:
            if start is not None or end is not None:
                return _PriceRequest(start=start, end=end, period=None, should_fetch=True)
            return _PriceRequest(start=None, end=None, period=period, should_fetch=True)

        latest_date = self.repository.latest_price_date(ticker, source="yfinance")
        if latest_date is None:
            if start is not None or end is not None:
                return _PriceRequest(start=start, end=end, period=None, should_fetch=True)
            return _PriceRequest(start=None, end=None, period=period, should_fetch=True)

        if start is None and end is None and latest_date >= self._latest_expected_trading_day():
            logger.info(
                "Incremental ingestion ticker=%s already current latest_stored_date=%s",
                ticker,
                latest_date,
            )
            return _PriceRequest(
                start=(latest_date + timedelta(days=1)).isoformat(),
                end=None,
                period=None,
                should_fetch=False,
            )

        next_date = latest_date + timedelta(days=1)
        effective_start = max(start_date, next_date) if start_date is not None else next_date
        if end_date is not None and effective_start >= end_date:
            return _PriceRequest(
                start=effective_start.isoformat(),
                end=end,
                period=None,
                should_fetch=False,
            )

        logger.info(
            "Incremental ingestion ticker=%s latest_stored_date=%s request_start=%s",
            ticker,
            latest_date,
            effective_start,
        )
        return _PriceRequest(
            start=effective_start.isoformat(),
            end=end,
            period=None,
            should_fetch=True,
        )

    def _validate_prices(self, prices: pd.DataFrame, *, incremental: bool) -> ValidationResult:
        if incremental and prices.empty:
            return ValidationResult(is_valid=True, errors=[], warnings=[], row_count=0)
        return self.validator.validate_daily_prices(prices)

    def _mark_failed_after_exception(
        self,
        *,
        ingestion_run_id: int | None,
        rows_fetched: int,
        rows_written: int,
        warning_count: int,
        error_count: int,
        error_message: str,
    ) -> None:
        if ingestion_run_id is None:
            return
        try:
            self.repository.rollback()
            self.repository.fail_ingestion_run(
                ingestion_run_id,
                error_message=error_message,
                rows_fetched=rows_fetched,
                rows_written=rows_written,
                warning_count=warning_count,
                error_count=error_count,
            )
        except Exception:
            logger.exception("Failed to mark ingestion run failed id=%s", ingestion_run_id)

    @classmethod
    def _summary(
        cls,
        *,
        ticker: str,
        status: str,
        rows_fetched: int,
        rows_written: int,
        warning_count: int,
        error_count: int,
        prices: pd.DataFrame,
        ingestion_run_id: int | None,
        message: str,
    ) -> IngestionSummary:
        earliest_date, latest_date = cls._price_date_bounds(prices)
        return IngestionSummary(
            ticker=ticker,
            status=status,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
            warning_count=warning_count,
            error_count=error_count,
            earliest_date=earliest_date,
            latest_date=latest_date,
            ingestion_run_id=ingestion_run_id,
            message=message,
        )

    @staticmethod
    def _price_date_bounds(prices: pd.DataFrame) -> tuple[date | None, date | None]:
        if prices.empty or "date" not in prices.columns:
            return None, None
        dates = pd.to_datetime(prices["date"], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.min().date(), dates.max().date()

    @staticmethod
    def _empty_price_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "dividends",
                "stock_splits",
                "source",
            ]
        )

    @staticmethod
    def _validation_message(
        metadata_result: ValidationResult,
        prices_result: ValidationResult,
    ) -> str:
        errors = metadata_result.errors + prices_result.errors
        return "Validation failed: " + "; ".join(errors)

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        if not isinstance(ticker, str) or not ticker.strip():
            raise MarketDataIngestionError("ticker must be a non-empty string")
        return ticker.strip().upper()

    @staticmethod
    def _parse_date(value: str, field_name: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise MarketDataIngestionError(
                f"{field_name} must be a valid ISO date string in YYYY-MM-DD format"
            ) from exc

    @staticmethod
    def _latest_expected_trading_day(today: date | None = None) -> date:
        current_day = today or date.today()
        if current_day.weekday() == 0:
            return current_day - timedelta(days=3)
        if current_day.weekday() == 6:
            return current_day - timedelta(days=2)
        if current_day.weekday() == 5:
            return current_day - timedelta(days=1)
        return current_day - timedelta(days=1)
