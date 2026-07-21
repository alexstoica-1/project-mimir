from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.database.models import Company, DailyPrice
from src.database.repository import MarketRepository


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as db_session:
        yield db_session


@pytest.fixture()
def repository(session: Session) -> MarketRepository:
    return MarketRepository(session)


def company_metadata(**overrides) -> dict[str, object]:
    metadata: dict[str, object] = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NMS",
        "quote_type": "EQUITY",
        "currency": "USD",
        "timezone": "EDT",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "website": "https://www.apple.com",
        "market_cap": 1_000_000,
        "source": "yfinance",
    }
    metadata.update(overrides)
    return metadata


def daily_prices() -> pd.DataFrame:
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


def test_company_insertion(repository: MarketRepository) -> None:
    company = repository.create_company(company_metadata(ticker="aapl"))

    assert company.ticker == "AAPL"
    assert repository.company_exists("AAPL")


def test_company_update(repository: MarketRepository) -> None:
    repository.create_company(company_metadata(company_name="Old Name"))

    company = repository.upsert_company(company_metadata(company_name="Apple Inc."))

    assert company.company_name == "Apple Inc."


def test_price_insertion(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())

    written = repository.insert_daily_prices(daily_prices())

    assert written == 2
    assert len(repository.get_prices("AAPL")) == 2


def test_duplicate_prevention(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())
    repository.insert_daily_prices(daily_prices())

    written = repository.insert_daily_prices(daily_prices())

    assert written == 0
    assert len(repository.get_prices("AAPL")) == 2


def test_upsert(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())
    repository.insert_daily_prices(daily_prices())
    updated = daily_prices()
    updated.loc[0, "close"] = 110.0

    written = repository.upsert_daily_prices(updated)

    prices = repository.get_prices("AAPL")
    assert written == 2
    assert str(prices[0].close) in {"110.0000000000", "110.000000000000000000"}


def test_latest_price_date(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())
    repository.insert_daily_prices(daily_prices())

    assert repository.latest_price_date("AAPL") == date(2024, 1, 3)


def test_retrieving_prices(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())
    repository.insert_daily_prices(daily_prices())

    prices = repository.get_prices("AAPL", start=date(2024, 1, 3))

    assert len(prices) == 1
    assert prices[0].date == date(2024, 1, 3)


def test_retrieving_company(repository: MarketRepository) -> None:
    repository.create_company(company_metadata())

    company = repository.get_company("aapl")

    assert company is not None
    assert company.company_name == "Apple Inc."


def test_failed_transaction_rollback(repository: MarketRepository, session: Session) -> None:
    repository.create_company(company_metadata())

    with pytest.raises(IntegrityError):
        repository.create_company(company_metadata())

    assert session.scalars(select(Company)).all()[0].ticker == "AAPL"
    assert repository.company_exists("AAPL")


def test_ingestion_run_creation(repository: MarketRepository) -> None:
    run = repository.create_ingestion_run(source="yfinance", ticker="aapl")

    assert run.id is not None
    assert run.ticker == "AAPL"
    assert run.status == "running"


def test_ingestion_run_completion(repository: MarketRepository) -> None:
    run = repository.create_ingestion_run(source="yfinance", ticker="AAPL")

    completed = repository.complete_ingestion_run(
        run.id,
        rows_fetched=2,
        rows_written=2,
        warning_count=1,
        error_count=0,
    )

    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert completed.rows_fetched == 2
    assert completed.rows_written == 2
    assert completed.warning_count == 1


def test_fail_ingestion_run(repository: MarketRepository) -> None:
    run = repository.create_ingestion_run(source="yfinance", ticker="AAPL")

    failed = repository.fail_ingestion_run(run.id, error_message="validation failed", error_count=1)

    assert failed.status == "failed"
    assert failed.error_message == "validation failed"
    assert failed.error_count == 1


def test_unique_price_constraint_exists(repository: MarketRepository, session: Session) -> None:
    repository.create_company(company_metadata())
    repository.insert_daily_prices(daily_prices())
    duplicate = DailyPrice(
        ticker="AAPL",
        date=date(2024, 1, 2),
        open=100,
        high=105,
        low=99,
        close=104,
        adjusted_close=103.5,
        volume=1000,
        dividends=0,
        stock_splits=0,
        source="yfinance",
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.commit()
