"""Train and track the volatility model comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.data import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    load_dataset,
)
from src.ml.evaluate import evaluate_prediction_frame, make_prediction_frame
from src.ml.garch_model import GarchForecaster, GarchPyfuncModel
from src.ml.lightgbm_model import LightGBMRegressorModel, train_local_lightgbm
from src.ml.lstm_model import train_lstm
from src.ml.registry import (
    log_common_metadata,
    log_prediction_artifacts,
    register_model_version,
    tracked_run,
    save_local_artifact,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime settings for a reproducible model comparison."""

    data_path: Path = Path("data/processed/v1/features_all_tickers.csv")
    output_dir: Path = Path("models/volatility")
    experiment_name: str = "mimir-volatility"
    tracking_uri: str = "sqlite:///mlflow.db"
    seed: int = 42
    tickers: tuple[str, ...] | None = None
    run_local_lightgbm: bool = True
    lstm_epochs: int = 30
    lstm_lookback: int = 60


def run_training(config: TrainingConfig) -> pd.DataFrame:
    """Train all requested models, evaluate them, and record MLflow runs."""

    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(config.data_path, tickers=config.tickers)
    splits = chronological_split(frame)
    dataset_hash = _file_hash(config.data_path)
    split_metadata = {
        "train_start": splits.train[DATE_COLUMN].min(),
        "train_end": splits.train[DATE_COLUMN].max(),
        "validation_start": splits.validation[DATE_COLUMN].min(),
        "validation_end": splits.validation[DATE_COLUMN].max(),
        "test_start": splits.test[DATE_COLUMN].min(),
        "test_end": splits.test[DATE_COLUMN].max(),
    }

    all_results: list[pd.DataFrame] = []
    with tracked_run(
        run_name="comparison-parent",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        tags={"model_scope": "comparison", "dataset_hash": dataset_hash},
    ) as mlflow:
        mlflow.log_params({
            "dataset_path": str(config.data_path),
            "dataset_hash": dataset_hash,
            "target": TARGET_COLUMN,
            "seed": config.seed,
            "row_count": len(frame),
            "ticker_count": frame["ticker"].nunique(),
        })
        log_common_metadata(
            mlflow,
            params={"dataset_path": str(config.data_path), "dataset_hash": dataset_hash},
            feature_columns=list(FEATURE_COLUMNS),
            split_metadata=split_metadata,
        )
        baseline = _baseline_results(splits.test)
        with tracked_run(
            run_name="baseline-naive",
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            nested=True,
        ) as child_mlflow:
            pooled, by_scope = evaluate_prediction_frame(baseline)
            log_common_metadata(child_mlflow, params={"baseline": "rv_20d"}, metrics=pooled)
            log_prediction_artifacts(child_mlflow, baseline, by_scope)
            all_results.append(_tag_metrics(by_scope, "baseline-naive"))

        all_results.extend(_train_garch(config, splits, split_metadata, dataset_hash))
        all_results.extend(_train_lightgbm(config, splits, split_metadata, dataset_hash))
        if config.run_local_lightgbm:
            all_results.extend(_train_local_lightgbm(config, splits, split_metadata, dataset_hash))
        all_results.extend(_train_lstm(config, frame, splits, split_metadata, dataset_hash))

        summary = pd.concat(all_results, ignore_index=True)
        mlflow.log_text(summary.to_csv(index=False), "comparison_summary.csv")

    summary_path = config.output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def _train_garch(
    config: TrainingConfig,
    splits: Any,
    split_metadata: dict[str, Any],
    dataset_hash: str,
) -> list[pd.DataFrame]:
    """Fit validation and final-test GARCH models per ticker."""

    validation_model = GarchForecaster().fit(splits.train)
    validation_predictions = _merge_model_predictions(
        validation_model.predict(splits.validation), splits.validation
    )
    pooled_validation, _ = evaluate_prediction_frame(validation_predictions)

    final_training = pd.concat([splits.train, splits.validation], ignore_index=True)
    test_model = GarchForecaster().fit(final_training)
    test_predictions = _merge_model_predictions(test_model.predict(splits.test), splits.test)
    pooled_test, by_scope_test = evaluate_prediction_frame(test_predictions)
    results = [_tag_metrics(by_scope_test, "garch")]

    with tracked_run(
        run_name="garch-parent",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        nested=True,
    ) as mlflow:
        log_common_metadata(
            mlflow,
            params={"model": "GARCH(1,1)", "distribution": "student_t", "horizon": 20},
            metrics={f"validation_{key}": value for key, value in pooled_validation.items()},
            feature_columns=["log_return"],
            split_metadata=split_metadata,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in pooled_test.items() if value == value})
        log_prediction_artifacts(mlflow, test_predictions, by_scope_test)
        mlflow.log_text(test_model.parameter_table().to_csv(index=False), "garch_parameters.csv")
        artifact = save_local_artifact(test_model, config.output_dir / "garch_models.joblib")
        mlflow.log_artifact(str(artifact))
        for ticker in sorted(test_model.parameters):
            ticker_validation = validation_predictions[validation_predictions["ticker"].eq(ticker)]
            ticker_test = test_predictions[test_predictions["ticker"].eq(ticker)]
            if ticker_test.empty:
                continue
            ticker_validation_metrics, ticker_validation_by_scope = evaluate_prediction_frame(
                ticker_validation
            )
            ticker_test_metrics, ticker_test_by_scope = evaluate_prediction_frame(ticker_test)
            with tracked_run(
                run_name=f"garch-{ticker}",
                tracking_uri=config.tracking_uri,
                experiment_name=config.experiment_name,
                nested=True,
                tags={"model_scope": "per-ticker", "ticker": ticker},
            ) as ticker_mlflow:
                log_common_metadata(
                    ticker_mlflow,
                    params={
                        "model": "garch",
                        "ticker": ticker,
                        "distribution": "student_t",
                        "horizon": 20,
                    },
                    metrics={
                        **{f"validation_{key}": value for key, value in ticker_validation_metrics.items()},
                        **{f"test_{key}": value for key, value in ticker_test_metrics.items()},
                    },
                    feature_columns=["log_return"],
                    split_metadata=split_metadata,
                )
                log_prediction_artifacts(ticker_mlflow, ticker_test, ticker_test_by_scope)
                ticker_mlflow.log_text(
                    test_model.parameter_table()
                    .query("ticker == @ticker")
                    .to_csv(index=False),
                    "garch_parameters.csv",
                )
                ticker_forecaster = GarchForecaster(horizon=test_model.horizon)
                ticker_forecaster.parameters = {ticker: test_model.parameters[ticker]}
                try:
                    garch_info = ticker_mlflow.pyfunc.log_model(
                        name=f"garch_model_{ticker}",
                        python_model=GarchPyfuncModel(ticker_forecaster),
                        input_example=ticker_test[["ticker", "date", "log_return"]].head(1),
                    )
                    register_model_version(
                        ticker_mlflow,
                        garch_info.model_uri,
                        f"mimir-garch-{ticker}",
                    )
                except Exception as exc:  # pragma: no cover - version-specific MLflow behavior
                    ticker_mlflow.log_text(str(exc), "garch_model_logging_warning.txt")
    return results


def _train_lightgbm(
    config: TrainingConfig,
    splits: Any,
    split_metadata: dict[str, Any],
    dataset_hash: str,
) -> list[pd.DataFrame]:
    """Fit validation-tuned pooled LightGBM and evaluate a final refit."""

    validation_model = LightGBMRegressorModel().fit(splits.train, splits.validation)
    validation_predictions = _tabular_predictions(validation_model, splits.validation)
    pooled_validation, _ = evaluate_prediction_frame(validation_predictions)

    combined = pd.concat([splits.train, splits.validation], ignore_index=True)
    final_params = dict(validation_model.params)
    best_iteration = getattr(validation_model.model, "best_iteration_", None)
    if best_iteration:
        final_params["n_estimators"] = int(best_iteration)
    test_model = LightGBMRegressorModel().fit(combined, None, params=final_params)
    test_predictions = _tabular_predictions(test_model, splits.test)
    pooled_test, by_scope_test = evaluate_prediction_frame(test_predictions)

    with tracked_run(
        run_name="lightgbm-global",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        nested=True,
    ) as mlflow:
        log_common_metadata(
            mlflow,
            params={"model": "lightgbm-global", **final_params},
            metrics={f"validation_{key}": value for key, value in pooled_validation.items()},
            feature_columns=list(FEATURE_COLUMNS),
            split_metadata=split_metadata,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in pooled_test.items() if value == value})
        log_prediction_artifacts(mlflow, test_predictions, by_scope_test)
        mlflow.log_text(test_model.feature_importance().to_csv(index=False), "feature_importance.csv")
        artifact = save_local_artifact(test_model, config.output_dir / "lightgbm_global.joblib")
        mlflow.log_artifact(str(artifact))
        try:
            import mlflow.lightgbm

            model_info = mlflow.lightgbm.log_model(test_model.model, name="lightgbm_model")
            register_model_version(
                mlflow,
                model_info.model_uri,
                "mimir-lightgbm-global",
            )
        except Exception as exc:  # pragma: no cover - version-specific MLflow behavior
            mlflow.log_text(str(exc), "lightgbm_model_logging_warning.txt")
    return [_tag_metrics(by_scope_test, "lightgbm-global")]


def _train_local_lightgbm(
    config: TrainingConfig,
    splits: Any,
    split_metadata: dict[str, Any],
    dataset_hash: str,
) -> list[pd.DataFrame]:
    """Train one LightGBM model per ticker for the optional comparison."""

    models = train_local_lightgbm(splits.train, splits.validation)
    combined = pd.concat([splits.train, splits.validation], ignore_index=True)
    results: list[pd.DataFrame] = []
    for ticker, validation_model in models.items():
        final_model = LightGBMRegressorModel().fit(
            combined[combined["ticker"].eq(ticker)],
            None,
            params={
                **validation_model.params,
                "n_estimators": int(getattr(validation_model.model, "best_iteration_", 500) or 500),
            },
        )
        test_rows = splits.test[splits.test["ticker"].eq(ticker)]
        if test_rows.empty:
            continue
        predictions = _tabular_predictions(final_model, test_rows)
        pooled, by_scope = evaluate_prediction_frame(predictions)
        with tracked_run(
            run_name=f"lightgbm-local-{ticker}",
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            nested=True,
        ) as mlflow:
            log_common_metadata(
                mlflow,
                params={"model": "lightgbm-local", "ticker": ticker},
                metrics=pooled,
                feature_columns=list(FEATURE_COLUMNS),
                split_metadata=split_metadata,
            )
            log_prediction_artifacts(mlflow, predictions, by_scope)
            artifact = save_local_artifact(
                final_model,
                config.output_dir / f"lightgbm_local_{ticker}.joblib",
            )
            mlflow.log_artifact(str(artifact))
            try:
                import mlflow.lightgbm

                local_info = mlflow.lightgbm.log_model(
                    final_model.model,
                    name=f"lightgbm_model_{ticker}",
                )
                register_model_version(
                    mlflow,
                    local_info.model_uri,
                    f"mimir-lightgbm-local-{ticker}",
                )
            except Exception as exc:  # pragma: no cover - version-specific MLflow behavior
                mlflow.log_text(str(exc), "lightgbm_model_logging_warning.txt")
        results.append(_tag_metrics(by_scope, f"lightgbm-local-{ticker}"))
    return results


def _train_lstm(
    config: TrainingConfig,
    frame: pd.DataFrame,
    splits: Any,
    split_metadata: dict[str, Any],
    dataset_hash: str,
) -> list[pd.DataFrame]:
    """Train, track, and evaluate the pooled PyTorch LSTM."""

    result = train_lstm(
        splits.train,
        splits.validation,
        frame,
        train_end=splits.train_end,
        validation_end=splits.validation_end,
        lookback=config.lstm_lookback,
        epochs=config.lstm_epochs,
        seed=config.seed,
    )
    validation_prediction, validation_metadata = result.predict(
        frame,
        start_after=splits.train_end,
    )
    validation_prediction_frame = _sequence_predictions(
        validation_metadata,
        validation_prediction,
        frame,
        end_at=splits.validation_end,
    )
    pooled_validation, _ = evaluate_prediction_frame(validation_prediction_frame)
    test_prediction, test_metadata = result.predict(frame, start_after=splits.validation_end)
    test_predictions = _sequence_predictions(test_metadata, test_prediction, frame)
    pooled_test, by_scope_test = evaluate_prediction_frame(test_predictions)

    with tracked_run(
        run_name="lstm-global",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        nested=True,
    ) as mlflow:
        log_common_metadata(
            mlflow,
            params={
                "model": "lstm-global",
                "lookback": config.lstm_lookback,
                "epochs": config.lstm_epochs,
                "device": result.device,
            },
            metrics={f"validation_{key}": value for key, value in pooled_validation.items()},
            feature_columns=list(FEATURE_COLUMNS),
            split_metadata=split_metadata,
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in pooled_test.items() if value == value})
        log_prediction_artifacts(mlflow, test_predictions, by_scope_test)
        mlflow.log_text(result.history.to_csv(index=False), "training_history.csv")
        artifact = result.save(config.output_dir / "lstm_global.pt")
        mlflow.log_artifact(str(artifact))
        try:
            import mlflow.pytorch

            model_info = mlflow.pytorch.log_model(
                result.model.network,
                name="lstm_model",
                input_example=np.zeros(
                    (1, config.lstm_lookback, result.model.input_size),
                    dtype=np.float32,
                ),
                serialization_format="pickle",
            )
            register_model_version(
                mlflow,
                model_info.model_uri,
                "mimir-lstm-global",
            )
        except Exception as exc:  # pragma: no cover - version-specific MLflow behavior
            mlflow.log_text(str(exc), "lstm_model_logging_warning.txt")
    return [_tag_metrics(by_scope_test, "lstm-global")]


def _baseline_results(test: pd.DataFrame) -> pd.DataFrame:
    """Create the no-learning persistence baseline."""

    return pd.DataFrame({
        "ticker": test["ticker"].to_numpy(),
        "date": test[DATE_COLUMN].to_numpy(),
        "actual": test[TARGET_COLUMN].to_numpy(dtype=float),
        "prediction": test["rv_20d"].to_numpy(dtype=float),
        "baseline": test["rv_20d"].to_numpy(dtype=float),
    })


def _tabular_predictions(model: LightGBMRegressorModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict a tabular frame and attach its naive baseline."""

    return pd.DataFrame({
        "ticker": frame["ticker"].to_numpy(),
        "date": frame[DATE_COLUMN].to_numpy(),
        "actual": frame[TARGET_COLUMN].to_numpy(dtype=float),
        "prediction": model.predict(frame),
        "baseline": frame["rv_20d"].to_numpy(dtype=float),
    })


def _merge_model_predictions(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach target and naive baseline to model output by ticker/date."""

    if predictions.empty:
        return pd.DataFrame(columns=["ticker", "date", "actual", "prediction", "baseline"])
    result = predictions.merge(
        truth[["ticker", DATE_COLUMN, TARGET_COLUMN, "rv_20d"]],
        on=["ticker", DATE_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    return result.rename(columns={TARGET_COLUMN: "actual", "rv_20d": "baseline"})


def _sequence_predictions(
    metadata: pd.DataFrame,
    prediction: np.ndarray,
    frame: pd.DataFrame,
    *,
    end_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Attach target and baseline to LSTM sequence predictions."""

    if metadata.empty:
        return pd.DataFrame(columns=["ticker", "date", "actual", "prediction", "baseline"])
    result = metadata.copy()
    result["prediction"] = prediction
    if end_at is not None:
        result = result[result[DATE_COLUMN] <= end_at]
    result = result.merge(
        frame[["ticker", DATE_COLUMN, TARGET_COLUMN, "rv_20d"]],
        on=["ticker", DATE_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    return result.rename(columns={TARGET_COLUMN: "actual", "rv_20d": "baseline"})[
        ["ticker", DATE_COLUMN, "actual", "prediction", "baseline"]
    ]


def _tag_metrics(metrics: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Add model identity to a metrics table."""

    result = metrics.copy()
    result.insert(0, "model", model_name)
    return result


def _file_hash(path: Path) -> str:
    """Hash the input dataset so MLflow runs identify their data."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(config: TrainingConfig | None = None) -> pd.DataFrame:
    """Convenience entrypoint for Python callers."""

    return run_training(config or TrainingConfig())
