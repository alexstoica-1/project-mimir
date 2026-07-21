"""Inspect yfinance collection output from the project root."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.collectors.yfinance_collector import YFinanceCollectionError, YFinanceCollector

logger = logging.getLogger(__name__)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    collector = YFinanceCollector()
    try:
        metadata = collector.fetch_company_metadata(args.ticker)
        prices = collector.fetch_daily_prices(
            args.ticker,
            start=args.start,
            end=args.end,
            period=args.period,
        )
    except YFinanceCollectionError as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1

    print_summary(metadata, prices)

    if args.save_raw:
        try:
            save_raw_outputs(metadata, prices)
        except OSError as exc:
            print(f"Failed to save raw yfinance outputs: {exc}", file=sys.stderr)
            return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect raw yfinance collection output.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to inspect.")
    parser.add_argument("--period", default="1y", help="yfinance period, ignored when start/end is set.")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format.")
    parser.add_argument("--save-raw", action="store_true", help="Save raw metadata and prices under data/raw.")
    return parser


def print_summary(metadata: dict[str, Any], prices: pd.DataFrame) -> None:
    print("\nMetadata")
    print(json.dumps(metadata, indent=2, default=json_default))

    print("\nPrice DataFrame")
    print(f"Shape: {prices.shape}")
    print(f"Columns: {list(prices.columns)}")

    print("\nDtypes")
    print(prices.dtypes)

    print("\nFirst five rows")
    print(prices.head(5))

    print("\nLast five rows")
    print(prices.tail(5))

    print("\nMissing values")
    print(prices.isna().sum())

    print(f"\nDuplicate rows: {int(prices.duplicated().sum())}")
    if prices.empty:
        print("Date range: <empty>")
    else:
        print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")


def save_raw_outputs(metadata: dict[str, Any], prices: pd.DataFrame) -> None:
    ticker = str(metadata["ticker"]).strip().upper()
    output_dir = Path("data") / "raw" / "yfinance"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / f"{ticker}_metadata.json"
    prices_path = output_dir / f"{ticker}_daily_prices.csv"

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    prices.to_csv(prices_path, index=False)

    logger.info("Saved metadata to %s", metadata_path)
    logger.info("Saved prices to %s", prices_path)


def json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
