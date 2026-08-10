"""Run batch predictions from locally saved forecasting artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.data import load_dataset
from src.ml.garch_model import GarchForecaster
from src.ml.lightgbm_model import LightGBMRegressorModel
from src.ml.lstm_model import LSTMTrainingResult


def build_parser() -> argparse.ArgumentParser:
    """Create the prediction CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["garch", "lightgbm", "lstm"], required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/v1/features_all_tickers.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictions.csv"))
    parser.add_argument("--tickers", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load one local artifact and write batch predictions."""

    args = build_parser().parse_args(argv)
    frame = load_dataset(args.data, tickers=args.tickers)
    if args.model == "lightgbm":
        model = LightGBMRegressorModel.load(str(args.artifact))
        predictions = pd.DataFrame({
            "ticker": frame["ticker"],
            "date": frame["date"],
            "prediction": model.predict(frame),
        })
    elif args.model == "lstm":
        model = LSTMTrainingResult.load(args.artifact)
        prediction_values, metadata = model.predict(frame)
        predictions = metadata.copy()
        predictions["prediction"] = prediction_values
    else:
        import joblib

        model = joblib.load(args.artifact)
        future_rows = pd.concat(
            [
                group[group["date"] > model.parameters[str(ticker)].train_end]
                for ticker, group in frame.groupby("ticker", sort=True)
                if str(ticker) in model.parameters
            ],
            ignore_index=True,
        )
        predictions = model.predict(future_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predictions: {len(predictions)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
