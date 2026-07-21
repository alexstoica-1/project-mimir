"""Repository layer for persisted market data."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from src.database.models import Company, DailyPrice, IngestionRun

logger = logging.getLogger(__name__)


class MarketRepository:
    """Transaction-owning repository for market data models."""

    PRICE_FIELDS = [
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
    COMPANY_FIELDS = [
        "ticker",
        "company_name",
        "exchange",
        "quote_type",
        "currency",
        "timezone",
        "sector",
        "industry",
        "country",
        "website",
        "market_cap",
        "source",
    ]

    def __init__(self, session: Session) -> None:
        """Create a repository using a caller-managed SQLAlchemy session."""

        self.session = session

    def create_company(self, metadata: Mapping[str, Any]) -> Company:
        """Insert one company and commit the transaction."""

        company = Company(**self._company_values(metadata))
        self.session.add(company)
        self._commit("create company")
        self.session.refresh(company)
        logger.info("Created company ticker=%s", company.ticker)
        return company

    def upsert_company(self, metadata: Mapping[str, Any], *, commit: bool = True) -> Company:
        """Insert a company or update its metadata when it already exists."""

        values = self._company_values(metadata)
        company = self.get_company(values["ticker"])
        if company is None:
            company = Company(**values)
            self.session.add(company)
        else:
            for key, value in values.items():
                setattr(company, key, value)

        if commit:
            self._commit("upsert company")
            self.session.refresh(company)
        logger.info("Upserted company ticker=%s", company.ticker)
        return company

    def insert_daily_prices(self, prices: pd.DataFrame | Sequence[Mapping[str, Any]]) -> int:
        """Insert new daily prices, skipping duplicate ticker/date/source rows."""

        rows = self._price_rows(prices)
        written = 0
        for row in rows:
            if self._get_price(row["ticker"], row["date"], row["source"]) is not None:
                continue
            self.session.add(DailyPrice(**row))
            written += 1

        self._commit("insert daily prices")
        logger.info("Inserted %s daily price rows", written)
        return written

    def upsert_daily_prices(
        self,
        prices: pd.DataFrame | Sequence[Mapping[str, Any]],
        *,
        commit: bool = True,
    ) -> int:
        """Insert daily prices or update existing rows by ticker/date/source."""

        rows = self._price_rows(prices)
        written = 0
        for row in rows:
            existing = self._get_price(row["ticker"], row["date"], row["source"])
            if existing is None:
                self.session.add(DailyPrice(**row))
            else:
                for key, value in row.items():
                    setattr(existing, key, value)
            written += 1

        if commit:
            self._commit("upsert daily prices")
        logger.info("Upserted %s daily price rows", written)
        return written

    def create_ingestion_run(
        self,
        *,
        source: str,
        ticker: str,
        status: str = "running",
        rows_fetched: int = 0,
        rows_written: int = 0,
        warning_count: int = 0,
        error_count: int = 0,
        error_message: str | None = None,
    ) -> IngestionRun:
        """Create an ingestion run audit record."""

        run = IngestionRun(
            source=source,
            ticker=ticker,
            started_at=datetime.now(UTC),
            status=status,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
            warning_count=warning_count,
            error_count=error_count,
            error_message=error_message,
        )
        self.session.add(run)
        self._commit("create ingestion run")
        self.session.refresh(run)
        logger.info("Created ingestion run id=%s ticker=%s source=%s", run.id, run.ticker, run.source)
        return run

    def complete_ingestion_run(
        self,
        run_id: int,
        *,
        rows_fetched: int | None = None,
        rows_written: int | None = None,
        warning_count: int | None = None,
        error_count: int | None = None,
    ) -> IngestionRun:
        """Mark an ingestion run as completed."""

        run = self._require_ingestion_run(run_id)
        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        if rows_fetched is not None:
            run.rows_fetched = rows_fetched
        if rows_written is not None:
            run.rows_written = rows_written
        if warning_count is not None:
            run.warning_count = warning_count
        if error_count is not None:
            run.error_count = error_count

        self._commit("complete ingestion run")
        self.session.refresh(run)
        logger.info("Completed ingestion run id=%s", run.id)
        return run

    def fail_ingestion_run(
        self,
        run_id: int,
        *,
        error_message: str,
        rows_fetched: int | None = None,
        rows_written: int | None = None,
        warning_count: int | None = None,
        error_count: int | None = None,
    ) -> IngestionRun:
        """Mark an ingestion run as failed."""

        run = self._require_ingestion_run(run_id)
        run.status = "failed"
        run.completed_at = datetime.now(UTC)
        run.error_message = error_message
        if rows_fetched is not None:
            run.rows_fetched = rows_fetched
        if rows_written is not None:
            run.rows_written = rows_written
        if warning_count is not None:
            run.warning_count = warning_count
        if error_count is not None:
            run.error_count = error_count

        self._commit("fail ingestion run")
        self.session.refresh(run)
        logger.info("Failed ingestion run id=%s", run.id)
        return run

    def company_exists(self, ticker: str) -> bool:
        """Return whether a company ticker exists."""

        return self.get_company(ticker) is not None

    def latest_price_date(self, ticker: str, source: str = "yfinance") -> date | None:
        """Return the latest stored daily price date for a ticker and source."""

        stmt = select(func.max(DailyPrice.date)).where(
            DailyPrice.ticker == self._normalize_ticker(ticker),
            DailyPrice.source == source,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_company(self, ticker: str) -> Company | None:
        """Return one company by ticker, or None when it is absent."""

        return self.session.get(Company, self._normalize_ticker(ticker))

    def get_prices(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> list[DailyPrice]:
        """Return daily prices ordered by date ascending."""

        stmt: Select[tuple[DailyPrice]] = select(DailyPrice).where(
            DailyPrice.ticker == self._normalize_ticker(ticker),
            DailyPrice.source == source,
        )
        if start is not None:
            stmt = stmt.where(DailyPrice.date >= self._as_date(start))
        if end is not None:
            stmt = stmt.where(DailyPrice.date <= self._as_date(end))

        stmt = stmt.order_by(DailyPrice.date.asc())
        return list(self.session.scalars(stmt).all())

    def _get_price(self, ticker: str, price_date: date, source: str) -> DailyPrice | None:
        stmt = select(DailyPrice).where(
            DailyPrice.ticker == self._normalize_ticker(ticker),
            DailyPrice.date == price_date,
            DailyPrice.source == source,
        )
        return self.session.scalars(stmt).one_or_none()

    def _require_ingestion_run(self, run_id: int) -> IngestionRun:
        run = self.session.get(IngestionRun, run_id)
        if run is None:
            raise ValueError(f"Ingestion run {run_id} does not exist")
        return run

    def _commit(self, operation: str) -> None:
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            logger.exception("Database transaction failed during %s", operation)
            raise

    def commit(self, operation: str = "repository operation") -> None:
        """Commit pending repository work and rollback if the commit fails."""

        self._commit(operation)

    def rollback(self) -> None:
        """Rollback pending repository work."""

        self.session.rollback()

    @classmethod
    def _company_values(cls, metadata: Mapping[str, Any]) -> dict[str, Any]:
        values = {field: metadata.get(field) for field in cls.COMPANY_FIELDS}
        values["ticker"] = cls._normalize_ticker(values["ticker"])
        return values

    @classmethod
    def _price_rows(
        cls,
        prices: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(prices, pd.DataFrame):
            raw_rows = prices.to_dict("records")
        else:
            raw_rows = list(prices)

        rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            row = {field: raw_row.get(field) for field in cls.PRICE_FIELDS}
            row["ticker"] = cls._normalize_ticker(row["ticker"])
            row["date"] = cls._as_date(row["date"]) # type: ignore
            for field in ("open", "high", "low", "close", "adjusted_close", "dividends", "stock_splits"):
                row[field] = cls._as_decimal(row[field])
            row["volume"] = int(row["volume"]) # type: ignore
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_ticker(ticker: Any) -> str:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ticker must be a non-empty string")
        return ticker.strip().upper()

    @staticmethod
    def _as_date(value: date | datetime | pd.Timestamp | str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, date):
            return value
        return pd.Timestamp(value).date()

    @staticmethod
    def _as_decimal(value: Any) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(value))
