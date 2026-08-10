"""Global and per-ticker LightGBM volatility regressors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ml.data import (
    DATE_COLUMN,
    TARGET_COLUMN,
    FeaturePreprocessor,
    inverse_target,
    transform_target,
)


@dataclass
class LightGBMRegressorModel:
    """LightGBM model plus its train-only feature preprocessing state."""

    model: Any = None
    preprocessor: FeaturePreprocessor | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame | None,
        *,
        params: dict[str, Any] | None = None,
    ) -> "LightGBMRegressorModel":
        """Fit a global model; preprocessing is fitted on training rows only."""

        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - setup failure
            raise RuntimeError("Install lightgbm before training LightGBM models") from exc

        self.preprocessor = FeaturePreprocessor.fit(train)
        train_x = self.preprocessor.transform(train)
        validation_x = self.preprocessor.transform(validation) if validation is not None else None
        self.params = {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            **(params or {}),
        }
        self.model = lgb.LGBMRegressor(**self.params)
        fit_kwargs: dict[str, Any] = {}
        if validation is not None and validation_x is not None:
            fit_kwargs.update({
                "eval_X": validation_x,
                "eval_y": transform_target(validation[TARGET_COLUMN]),
                "eval_names": ["validation"],
                "callbacks": [lgb.early_stopping(50, verbose=False)],
            })
        self.model.fit(train_x, transform_target(train[TARGET_COLUMN]), **fit_kwargs)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict non-negative raw decimal volatility."""

        if self.model is None or self.preprocessor is None:
            raise RuntimeError("LightGBM model has not been fitted")
        transformed = self.model.predict(self.preprocessor.transform(frame))
        return inverse_target(transformed)

    def feature_importance(self) -> pd.DataFrame:
        """Return gain-based feature importance."""

        if self.model is None or self.preprocessor is None:
            raise RuntimeError("LightGBM model has not been fitted")
        return (
            pd.DataFrame({
                "feature": self.preprocessor.output_columns,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        """Save the wrapper, preprocessing state, and native model with joblib."""

        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMRegressorModel":
        """Load a saved wrapper."""

        import joblib

        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not a LightGBMRegressorModel")
        return loaded


def train_local_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, LightGBMRegressorModel]:
    """Train one LightGBM model per ticker using the shared model settings."""

    models: dict[str, LightGBMRegressorModel] = {}
    for ticker, ticker_train in train.groupby("ticker", sort=True):
        ticker_validation = validation[validation["ticker"].eq(ticker)]
        if ticker_validation.empty:
            continue
        models[str(ticker)] = LightGBMRegressorModel().fit(
            ticker_train,
            ticker_validation,
            params=params,
        )
    return models


def prediction_frame_for_model(
    model: LightGBMRegressorModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build prediction rows using the model's raw-volatility output."""

    return pd.DataFrame({
        "ticker": frame["ticker"].to_numpy(),
        DATE_COLUMN: frame[DATE_COLUMN].to_numpy(),
        "actual": frame[TARGET_COLUMN].to_numpy(dtype=float),
        "prediction": model.predict(frame),
        "baseline": frame["rv_20d"].to_numpy(dtype=float),
    })
