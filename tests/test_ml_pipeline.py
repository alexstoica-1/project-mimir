"""Fast deterministic tests for forecasting data, models, and tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS
from src.ml.data import (
    TARGET_COLUMN,
    FeaturePreprocessor,
    chronological_split,
    inverse_target,
    load_dataset,
    make_latest_sequence,
    make_sequence_arrays,
    transform_target,
)
from src.ml.evaluate import evaluate_prediction_frame, regression_metrics
from src.ml.garch_model import GarchForecaster
from src.ml.lightgbm_model import LightGBMRegressorModel
from src.ml.lstm_model import train_lstm
from src.ml.registry import tracked_run


def synthetic_model_frame(periods: int = 150) -> pd.DataFrame:
    """Create two ticker histories with all required causal model columns."""

    dates = pd.bdate_range("2020-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 0.002)):
        steps = np.arange(periods, dtype=float)
        returns = 0.01 * np.sin(steps / 5.0 + offset) + 0.002 * np.cos(steps / 3.0)
        rv = 0.15 + 0.02 * np.sin(steps / 9.0 + offset)
        for index, row_date in enumerate(dates):
            row = {
                "ticker": ticker,
                "date": row_date,
                "log_return": returns[index],
                "rv_20d": rv[index],
                TARGET_COLUMN: rv[min(index + 20, periods - 1)],
            }
            for column in FEATURE_COLUMNS:
                row.setdefault(column, float(0.01 * (index + 1)))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_chronological_split_has_disjoint_date_ranges() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)

    assert splits.train["date"].max() < splits.validation["date"].min()
    assert splits.validation["date"].max() < splits.test["date"].min()
    assert set(splits.train["date"]).isdisjoint(splits.validation["date"])
    assert set(splits.validation["date"]).isdisjoint(splits.test["date"])


def test_loader_rejects_target_or_feature_missing(tmp_path) -> None:
    path = tmp_path / "features.csv"
    synthetic_model_frame(40).drop(columns="rv_20d").to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_target_transform_round_trip_and_preprocessor_excludes_target() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    transformed = transform_target(frame[TARGET_COLUMN])

    assert np.allclose(inverse_target(transformed), frame[TARGET_COLUMN])
    preprocessor = FeaturePreprocessor.fit(splits.train)
    transformed_frame = preprocessor.transform(splits.validation)
    assert TARGET_COLUMN not in transformed_frame.columns
    assert "ticker_AAA" in transformed_frame.columns


def test_metrics_and_prediction_frame_include_per_ticker_results() -> None:
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "date": pd.date_range("2024-01-01", periods=4),
        "actual": [1.0, 2.0, 1.0, 2.0],
        "prediction": [1.1, 1.9, 0.9, 2.1],
        "baseline": [1.5, 1.5, 1.5, 1.5],
    })
    pooled, by_scope = evaluate_prediction_frame(frame)

    assert pooled["mae"] == pytest.approx(0.1)
    assert set(by_scope["scope"]) == {"pooled", "AAA", "BBB"}
    assert regression_metrics([1, 1], [1, 2])["pearson"] != regression_metrics([1, 1], [1, 2])["pearson"]


def test_lstm_sequences_are_ticker_isolated() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    preprocessor = FeaturePreprocessor.fit(splits.train)
    x, y, metadata = make_sequence_arrays(
        frame,
        preprocessor,
        lookback=10,
        start_after=splits.train_end,
        end_at=splits.validation_end,
    )

    assert x.shape[0] == len(y) == len(metadata)
    assert x.shape[1] == 10
    assert set(metadata["ticker"]) == {"AAA", "BBB"}
    assert not metadata.duplicated(["ticker", "date"]).any()


def test_lstm_live_sequence_is_target_free_and_ticker_isolated() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    preprocessor = FeaturePreprocessor.fit(splits.train)
    live_frame = frame[frame["ticker"].eq("AAA")].drop(columns=TARGET_COLUMN)
    sequence, metadata = make_latest_sequence(live_frame, preprocessor, lookback=10)

    assert TARGET_COLUMN not in live_frame.columns
    assert sequence.shape == (1, 10, len(preprocessor.output_columns))
    assert metadata.to_dict("records") == [{"ticker": "AAA", "date": live_frame["date"].iloc[-1]}]
    with pytest.raises(ValueError, match="exactly one ticker"):
        make_latest_sequence(frame.drop(columns=TARGET_COLUMN), preprocessor, lookback=10)


def test_lightgbm_smoke_fit_and_predict() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    model = LightGBMRegressorModel().fit(
        splits.train,
        splits.validation,
        params={"n_estimators": 10, "num_leaves": 7, "verbosity": -1},
    )
    predictions = model.predict(splits.test)

    assert len(predictions) == len(splits.test)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_lightgbm_save_and_load_preserves_predictions(tmp_path) -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    model = LightGBMRegressorModel().fit(
        splits.train,
        splits.validation,
        params={"n_estimators": 5, "num_leaves": 7, "verbosity": -1},
    )
    path = tmp_path / "lightgbm.joblib"
    model.save(str(path))

    assert np.allclose(model.predict(splits.test), LightGBMRegressorModel.load(str(path)).predict(splits.test))


def test_garch_smoke_fit_and_forecast() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    model = GarchForecaster(min_train_rows=50).fit(splits.train)
    predictions = model.predict(splits.test)

    assert not predictions.empty
    assert np.isfinite(predictions["prediction"]).all()
    assert (predictions["prediction"] >= 0).all()
    assert predictions["date"].min() >= splits.test["date"].min()


def test_lstm_smoke_fit_and_predict() -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    result = train_lstm(
        splits.train,
        splits.validation,
        frame,
        train_end=splits.train_end,
        validation_end=splits.validation_end,
        lookback=10,
        hidden_size=8,
        num_layers=1,
        epochs=1,
        batch_size=32,
    )
    predictions, metadata = result.predict(frame, start_after=splits.validation_end)

    assert len(predictions) == len(metadata) > 0
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    live_prediction = result.predict_latest(
        frame[frame["ticker"].eq("AAA")].drop(columns=TARGET_COLUMN)
    )
    assert np.isfinite(live_prediction)
    assert live_prediction >= 0


def test_lstm_save_and_load_preserves_predictions(tmp_path) -> None:
    frame = synthetic_model_frame()
    splits = chronological_split(frame)
    result = train_lstm(
        splits.train,
        splits.validation,
        frame,
        train_end=splits.train_end,
        validation_end=splits.validation_end,
        lookback=10,
        hidden_size=8,
        num_layers=1,
        epochs=1,
        batch_size=32,
    )
    original_predictions, original_metadata = result.predict(frame, start_after=splits.validation_end)
    path = result.save(tmp_path / "lstm.pt")
    loaded_predictions, loaded_metadata = result.load(path).predict(
        frame,
        start_after=splits.validation_end,
    )

    assert original_metadata.equals(loaded_metadata)
    assert np.allclose(original_predictions, loaded_predictions)
    live_frame = frame[frame["ticker"].eq("AAA")].drop(columns=TARGET_COLUMN)
    assert result.predict_latest(live_frame) == pytest.approx(result.load(path).predict_latest(live_frame))


def test_mlflow_smoke_run(tmp_path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with tracked_run(
        run_name="test-run",
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
    ) as mlflow:
        mlflow.log_param("purpose", "unit-test")
        mlflow.log_metric("mae", 0.1)

    runs = mlflow.search_runs(experiment_names=["test-experiment"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.purpose"] == "unit-test"
