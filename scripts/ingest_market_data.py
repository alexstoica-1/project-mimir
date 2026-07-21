"""Run market data ingestion from the command line."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable

from src.collectors.yfinance_collector import YFinanceCollector
from src.database.connection import SessionLocal
from src.database.repository import MarketRepository
from src.services import IngestionSummary, MarketDataIngestionError, MarketDataIngestionService
from src.validation import YFinanceDataValidator

logger = logging.getLogger(__name__)


def main() -> int:
    return main_args(sys.argv[1:])


def main_args(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    tickers = args.tickers if args.tickers else [args.ticker]
    summaries: list[IngestionSummary] = []
    failures = 0

    for ticker in tickers:
        try:
            summary = ingest_one(
                ticker=ticker,
                period=args.period,
                start=args.start,
                end=args.end,
                incremental=not args.full_refresh,
            )
            summaries.append(summary)
            print_summary(summary)
            if summary.status != "completed":
                failures += 1
                if not args.continue_on_error:
                    break
        except MarketDataIngestionError as exc:
            failures += 1
            logger.exception("Ingestion failed for ticker=%s", ticker)
            print(f"\n{ticker.strip().upper()}")
            print(f"Status: failed")
            print(f"Message: {exc}")
            if not args.continue_on_error:
                break

    print_totals(summaries=summaries, failures=failures, requested_tickers=tickers)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest yfinance market data.")
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--ticker", help="One ticker to ingest.")
    ticker_group.add_argument("--tickers", nargs="+", help="One or more tickers to ingest.")
    parser.add_argument("--period", default="5y", help="yfinance period for non-date requests.")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format.")
    parser.add_argument("--full-refresh", action="store_true", help="Disable incremental ingestion.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later tickers after a ticker fails.",
    )
    return parser


def ingest_one(
    *,
    ticker: str,
    period: str,
    start: str | None,
    end: str | None,
    incremental: bool,
) -> IngestionSummary:
    with SessionLocal() as session:
        service = MarketDataIngestionService(
            collector=YFinanceCollector(),
            validator=YFinanceDataValidator(),
            repository=MarketRepository(session),
        )
        return service.ingest_ticker(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            incremental=incremental,
        )


def print_summary(summary: IngestionSummary) -> None:
    print(f"\n{summary.ticker}")
    print(f"Status: {summary.status}")
    print(f"Rows fetched: {summary.rows_fetched}")
    print(f"Rows written: {summary.rows_written}")
    print(f"Warnings: {summary.warning_count}")
    print(f"Errors: {summary.error_count}")
    print(f"Earliest date: {summary.earliest_date}")
    print(f"Latest date: {summary.latest_date}")
    print(f"Ingestion run ID: {summary.ingestion_run_id}")
    print(f"Message: {summary.message}")


def print_totals(
    *,
    summaries: list[IngestionSummary],
    failures: int,
    requested_tickers: Iterable[str],
) -> None:
    requested_count = len(list(requested_tickers))
    succeeded = sum(1 for summary in summaries if summary.status == "completed")
    rows_fetched = sum(summary.rows_fetched for summary in summaries)
    rows_written = sum(summary.rows_written for summary in summaries)
    print("\nTotals")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failures}")
    print(f"Rows fetched: {rows_fetched}")
    print(f"Rows written: {rows_written}")
    if succeeded + failures < requested_count:
        print(f"Skipped: {requested_count - succeeded - failures}")


if __name__ == "__main__":
    raise SystemExit(main())
