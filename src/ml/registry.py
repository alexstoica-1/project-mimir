"""Small MLflow tracking and artifact helpers."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "mimir-volatility",
) -> Any:
    """Configure MLflow and return the active experiment."""

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised in setup failures
        raise RuntimeError("Install mlflow before running model training") from exc

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


@contextmanager
def tracked_run(
    *,
    run_name: str,
    tracking_uri: str | None = None,
    experiment_name: str = "mimir-volatility",
    nested: bool = False,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Create a configured MLflow run and yield the active module."""

    mlflow = configure_mlflow(tracking_uri, experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
        yield mlflow


def log_common_metadata(
    mlflow: Any,
    *,
    params: dict[str, Any],
    metrics: dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> None:
    """Log JSON-safe parameters and common dataset metadata."""

    safe_params = {key: _json_scalar(value) for key, value in params.items()}
    if feature_columns is not None:
        safe_params["feature_count"] = len(feature_columns)
    if split_metadata:
        safe_params.update({key: _json_scalar(value) for key, value in split_metadata.items()})
    mlflow.log_params(safe_params)
    if metrics:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if value == value})
    if feature_columns is not None:
        mlflow.log_text(json.dumps(feature_columns, indent=2), "feature_columns.json")


def log_prediction_artifacts(mlflow: Any, predictions: Any, metrics_by_scope: Any) -> None:
    """Log prediction and metric tables without requiring caller-managed temp files."""

    mlflow.log_text(predictions.to_csv(index=False), "predictions.csv")
    mlflow.log_text(metrics_by_scope.to_csv(index=False), "metrics_by_scope.csv")


def log_json_artifact(mlflow: Any, payload: dict[str, Any], artifact_file: str) -> None:
    """Log a JSON artifact directly."""

    mlflow.log_text(json.dumps(payload, indent=2, default=str), artifact_file)


def save_local_artifact(payload: Any, path: str | Path) -> Path:
    """Serialize a Python object for local inference alongside MLflow artifacts."""

    import joblib

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)
    return artifact_path


def register_model_version(
    mlflow: Any,
    model_uri: str,
    registered_name: str,
    *,
    alias: str | None = None,
) -> Any:
    """Register a logged MLflow model and optionally assign an alias."""

    version = mlflow.register_model(model_uri=model_uri, name=registered_name)
    if alias:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(registered_name, alias, version.version)
    return version


def _json_scalar(value: Any) -> Any:
    """Convert common NumPy/Pandas scalars to MLflow-safe values."""

    if hasattr(value, "item"):
        return value.item()
    return value
