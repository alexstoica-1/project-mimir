"""Exercise collector, validator, repository, and PostgreSQL together."""

from __future__ import annotations

import logging
import sys

from src.collectors.yfinance_collector import YFinanceCollectionError, YFinanceCollector
from src.database.connection import SessionLocal
from src.database.repository import MarketRepository
from src.validation import YFinanceDataValidator

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    collector = YFinanceCollector()
    validator = YFinanceDataValidator()

    try:
        metadata = collector.fetch_company_metadata("AAPL")
        prices = collector.fetch_daily_prices("AAPL", period="1y")
    except YFinanceCollectionError as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1

    metadata_result = validator.validate_company_metadata(metadata)
    prices_result = validator.validate_daily_prices(prices)
    if not metadata_result.is_valid or not prices_result.is_valid:
        print("Validation failed.", file=sys.stderr)
        for error in metadata_result.errors + prices_result.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    with SessionLocal() as session:
        repository = MarketRepository(session)
        repository.upsert_company(metadata)
        rows_written = repository.upsert_daily_prices(prices)
        stored_company = repository.get_company("AAPL")
        stored_prices = repository.get_prices("AAPL")

    if stored_company is None:
        print("Company was not stored.", file=sys.stderr)
        return 1

    print(f"Company stored: {stored_company.ticker} - {stored_company.company_name}")
    print(f"Number of rows: {len(stored_prices)}")
    if stored_prices:
        print(f"Earliest date: {stored_prices[0].date}")
        print(f"Latest date: {stored_prices[-1].date}")
    print(f"Rows written or updated: {rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
