"""Helpers for loading locally saved forecasting artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ml.lightgbm_model import LightGBMRegressorModel
from src.ml.lstm_model import LSTMTrainingResult


def load_lightgbm(path: str | Path) -> LightGBMRegressorModel:
    """Load a locally saved LightGBM wrapper."""

    return LightGBMRegressorModel.load(str(path))


def load_lstm(path: str | Path) -> LSTMTrainingResult:
    """Load a locally saved PyTorch LSTM wrapper."""

    return LSTMTrainingResult.load(path)


def predict_lightgbm(path: str | Path, frame: Any) -> Any:
    """Load a LightGBM artifact and predict raw decimal volatility."""

    return load_lightgbm(path).predict(frame)


def predict_lstm(path: str | Path, frame: Any) -> Any:
    """Load an LSTM artifact and predict sequence forecasts."""

    return load_lstm(path).predict(frame)
