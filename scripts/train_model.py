"""Train and track GARCH, LightGBM, and LSTM volatility models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make both ``python -m scripts.train_model`` and ``python scripts/train_model.py``
# work when the repository has not been installed as a package.
if __package__ in {None, ""}:  # pragma: no cover - exercised by CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.train import TrainingConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the model-training CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/v1/features_all_tickers.csv"),
        help="Model-ready feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/volatility"),
        help="Directory for local model artifacts and summaries.",
    )
    parser.add_argument(
        "--experiment",
        default="mimir-volatility",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional ticker subset for smoke tests or focused experiments.",
    )
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--lstm-lookback", type=int, default=60)
    parser.add_argument(
        "--no-local-lightgbm",
        action="store_true",
        help="Skip the optional per-ticker LightGBM comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run training and print the comparison summary."""

    args = build_parser().parse_args(argv)
    summary = run_training(
        TrainingConfig(
            data_path=args.data,
            output_dir=args.output_dir,
            experiment_name=args.experiment,
            tracking_uri=args.tracking_uri,
            seed=args.seed,
            tickers=tuple(args.tickers) if args.tickers else None,
            run_local_lightgbm=not args.no_local_lightgbm,
            lstm_epochs=args.lstm_epochs,
            lstm_lookback=args.lstm_lookback,
        )
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
