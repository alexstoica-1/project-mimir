"""Build model-ready volatility features from persisted market data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.database.connection import SessionLocal
from src.database.repository import MarketRepository
from src.features import MarketFeaturePipeline

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH = Path("data/processed/v1/features_all_tickers.csv")


def main() -> int:
    """Run the command-line feature pipeline."""

    return main_args(sys.argv[1:])


def main_args(argv: list[str]) -> int:
    """Parse feature-build arguments and return a process exit status."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    tickers = args.tickers if args.tickers else ([args.ticker] if args.ticker else None)

    try:
        with SessionLocal() as session:
            pipeline = MarketFeaturePipeline(MarketRepository(session))
            features = pipeline.build_for_universe(
                tickers=tickers,
                start=args.start,
                end=args.end,
                source=args.source,
            )
        output_path = pipeline.save_csv(features, args.output)
    except Exception:
        logger.exception("Feature build failed")
        return 1

    print(f"Tickers processed: {features['ticker'].nunique()}")
    print(f"Feature rows: {len(features)}")
    print(f"Output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the feature-build CLI parser."""

    parser = argparse.ArgumentParser(description="Build features from persisted daily market prices.")
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker", help="Build features for one ticker.")
    ticker_group.add_argument("--tickers", nargs="+", help="Build features for selected tickers.")
    parser.add_argument("--start", help="Optional inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Optional inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--source", default="yfinance", help="Persisted market-data source.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
