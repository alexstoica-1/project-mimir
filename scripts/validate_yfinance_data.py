"""Collect and validate yfinance data from the project root."""

from __future__ import annotations

import argparse
import logging
import sys

from src.collectors.yfinance_collector import YFinanceCollectionError, YFinanceCollector
from src.validation import ValidationResult, YFinanceDataValidator

logger = logging.getLogger(__name__)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    collector = YFinanceCollector()
    validator = YFinanceDataValidator()

    try:
        metadata = collector.fetch_company_metadata(args.ticker)
        prices = collector.fetch_daily_prices(args.ticker, period=args.period)
    except YFinanceCollectionError as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1

    metadata_result = validator.validate_company_metadata(metadata)
    prices_result = validator.validate_daily_prices(prices)

    print_result("Company Metadata", metadata_result)
    print_result("Daily Prices", prices_result)

    if not metadata_result.is_valid or not prices_result.is_valid:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate yfinance collector output.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to validate.")
    parser.add_argument("--period", default="1y", help="yfinance price period to validate.")
    return parser


def print_result(label: str, result: ValidationResult) -> None:
    print(f"\n{label}")
    print(f"Valid: {result.is_valid}")
    print(f"Rows: {result.row_count}")

    print("Errors:")
    if result.errors:
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("  - None")

    print("Warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"  - {warning}")
    else:
        print("  - None")


if __name__ == "__main__":
    raise SystemExit(main())
