"""SQLAlchemy ORM models for market data storage."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from src.database.base import Base


class TimestampMixin:
    """Common creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Company(TimestampMixin, Base):
    """Company metadata collected from a market data source."""

    __tablename__ = "companies"

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    exchange: Mapped[Optional[str]] = mapped_column(String)
    quote_type: Mapped[Optional[str]] = mapped_column(String)
    currency: Mapped[Optional[str]] = mapped_column(String)
    timezone: Mapped[Optional[str]] = mapped_column(String)
    sector: Mapped[Optional[str]] = mapped_column(String)
    industry: Mapped[Optional[str]] = mapped_column(String)
    country: Mapped[Optional[str]] = mapped_column(String)
    website: Mapped[Optional[str]] = mapped_column(String)
    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)

    daily_prices: Mapped[list[DailyPrice]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    @validates("ticker")
    def _uppercase_ticker(self, _key: str, value: str) -> str:
        if value is None:
            raise ValueError("ticker cannot be None")
        return value.strip().upper()


class DailyPrice(TimestampMixin, Base):
    """One normalized daily OHLCV price row."""

    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("ticker", "date", "source", name="uq_daily_prices_ticker_date_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(
        String,
        ForeignKey("companies.ticker"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    adjusted_close: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    dividends: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    stock_splits: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)

    company: Mapped[Company] = relationship(back_populates="daily_prices")

    @validates("ticker")
    def _uppercase_ticker(self, _key: str, value: str) -> str:
        if value is None:
            raise ValueError("ticker cannot be None")
        return value.strip().upper()


class IngestionRun(Base):
    """Audit record for one data ingestion attempt."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    rows_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    @validates("ticker")
    def _uppercase_ticker(self, _key: str, value: str) -> str:
        if value is None:
            raise ValueError("ticker cannot be None")
        return value.strip().upper()
