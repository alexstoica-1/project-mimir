"""Evaluation metrics and prediction tables for volatility forecasts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    """Return a finite correlation or NaN when a series is constant/empty."""

    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def regression_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    baseline: Iterable[float] | None = None,
) -> dict[str, float]:
    """Calculate common raw-volatility regression metrics."""

    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.maximum(0.0, np.asarray(list(predicted), dtype=float))
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_series = pd.Series(actual_array[valid])
    predicted_series = pd.Series(predicted_array[valid])
    if actual_series.empty:
        return {name: float("nan") for name in (
            "mae", "rmse", "r2", "pearson", "spearman", "baseline_mae"
        )}

    metrics = {
        "mae": float(mean_absolute_error(actual_series, predicted_series)),
        "rmse": float(np.sqrt(mean_squared_error(actual_series, predicted_series))),
        "r2": float(r2_score(actual_series, predicted_series)) if len(actual_series) > 1 else float("nan"),
        "pearson": _safe_correlation(actual_series, predicted_series, "pearson"),
        "spearman": _safe_correlation(actual_series, predicted_series, "spearman"),
    }
    if baseline is not None:
        baseline_array = np.asarray(list(baseline), dtype=float)[valid]
        metrics["baseline_mae"] = float(mean_absolute_error(actual_series, baseline_array))
        metrics["baseline_rmse"] = float(np.sqrt(mean_squared_error(actual_series, baseline_array)))
        metrics["model_vs_baseline_pearson"] = _safe_correlation(
            pd.Series(baseline_array), predicted_series, "pearson"
        )
    return metrics


def evaluate_prediction_frame(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate pooled and per-ticker predictions from a standard frame."""

    required = {"ticker", "date", "actual", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {', '.join(missing)}")
    baseline = predictions.get("baseline")
    pooled = regression_metrics(
        predictions["actual"],
        predictions["prediction"],
        baseline,
    )
    rows = [{"scope": "pooled", **pooled}]
    for ticker, group in predictions.groupby("ticker", sort=True):
        group_baseline = group["baseline"] if "baseline" in group else None
        rows.append({
            "scope": str(ticker),
            **regression_metrics(group["actual"], group["prediction"], group_baseline),
        })
    return pooled, pd.DataFrame(rows)


def make_prediction_frame(
    metadata: pd.DataFrame,
    actual: Iterable[float],
    prediction: Iterable[float],
    baseline: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Build a consistent prediction artifact for MLflow and analysis."""

    result = metadata[["ticker", "date"]].copy().reset_index(drop=True)
    result["actual"] = np.asarray(list(actual), dtype=float)
    result["prediction"] = np.maximum(0.0, np.asarray(list(prediction), dtype=float))
    if baseline is not None:
        result["baseline"] = np.asarray(list(baseline), dtype=float)
    return result.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
