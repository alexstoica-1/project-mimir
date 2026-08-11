"""Shared data preparation for chronological volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.pipeline import FEATURE_COLUMNS

TARGET_COLUMN = "target_rv_20d"
TICKER_COLUMN = "ticker"
DATE_COLUMN = "date"
DEFAULT_LOOKBACK = 60


@dataclass(frozen=True)
class DateSplits:
    """Chronological train/validation/test frames and their date boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class FeaturePreprocessor:
    """Train-only numeric scaling and ticker one-hot encoding."""

    feature_columns: list[str]
    scaler: StandardScaler
    ticker_encoder: OneHotEncoder
    output_columns: list[str]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit preprocessing state using training rows only."""

        numeric = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(numeric)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            train[[TICKER_COLUMN]]
        )
        ticker_columns = [f"ticker_{value}" for value in encoder.categories_[0]]
        return cls(
            feature_columns=list(FEATURE_COLUMNS),
            scaler=scaler,
            ticker_encoder=encoder,
            output_columns=[*FEATURE_COLUMNS, *ticker_columns],
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform features without refitting any state."""

        numeric = self.scaler.transform(frame.loc[:, self.feature_columns].to_numpy(dtype=np.float64))
        tickers = self.ticker_encoder.transform(frame[[TICKER_COLUMN]])
        values = np.column_stack([numeric, tickers])
        return pd.DataFrame(values, index=frame.index, columns=self.output_columns)


def load_dataset(path: str | Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Load and validate the model-ready feature CSV."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Feature dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path, parse_dates=[DATE_COLUMN])
    required = {TICKER_COLUMN, DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {', '.join(missing)}")

    result = frame.copy()
    result[TICKER_COLUMN] = result[TICKER_COLUMN].astype(str).str.strip().str.upper()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise")
    if result.duplicated([TICKER_COLUMN, DATE_COLUMN]).any():
        raise ValueError("Feature dataset contains duplicate ticker/date rows")

    if tickers is not None:
        selected = {ticker.strip().upper() for ticker in tickers}
        result = result[result[TICKER_COLUMN].isin(selected)].copy()

    numeric_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature dataset contains non-finite model values")
    if (result[TARGET_COLUMN] < 0).any():
        raise ValueError("target_rv_20d must be non-negative")
    if result.empty:
        raise ValueError("Feature dataset contains no rows after ticker filtering")
    return result.sort_values([TICKER_COLUMN, DATE_COLUMN], kind="stable").reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DateSplits:
    """Split all tickers by shared calendar-date cutoffs without shuffling."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test period")

    dates = pd.DatetimeIndex(frame[DATE_COLUMN].drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for a split")
    train_index = max(1, min(len(dates) - 2, int(len(dates) * train_fraction) - 1))
    validation_index = max(
        train_index + 1,
        min(len(dates) - 1, int(len(dates) * (train_fraction + validation_fraction)) - 1),
    )
    train_end = dates[train_index]
    validation_end = dates[validation_index]

    train = frame[frame[DATE_COLUMN] <= train_end].copy()
    validation = frame[
        (frame[DATE_COLUMN] > train_end) & (frame[DATE_COLUMN] <= validation_end)
    ].copy()
    test = frame[frame[DATE_COLUMN] > validation_end].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return DateSplits(train, validation, test, train_end, validation_end)


def transform_target(values: pd.Series | np.ndarray) -> np.ndarray:
    """Transform non-negative volatility targets for ML regressors."""

    return np.log1p(np.asarray(values, dtype=np.float64))


def inverse_target(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert transformed predictions back to decimal volatility."""

    return np.maximum(0.0, np.expm1(np.asarray(values, dtype=np.float64)))


def make_sequence_arrays(
    frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    *,
    lookback: int,
    start_after: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create ticker-isolated sequences ending in an optional date interval."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    transformed = preprocessor.transform(frame)
    work = frame[[TICKER_COLUMN, DATE_COLUMN, TARGET_COLUMN]].copy()
    work["_row_position"] = np.arange(len(work))
    work = work.sort_values([TICKER_COLUMN, DATE_COLUMN], kind="stable")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, object]] = []
    for ticker, ticker_rows in work.groupby(TICKER_COLUMN, sort=False):
        positions = ticker_rows["_row_position"].to_numpy(dtype=int)
        dates = ticker_rows[DATE_COLUMN].to_numpy()
        values = transformed.iloc[positions].to_numpy(dtype=np.float32)
        target_values = ticker_rows[TARGET_COLUMN].to_numpy(dtype=float)
        for end_position in range(lookback - 1, len(ticker_rows)):
            end_date = pd.Timestamp(dates[end_position])
            if start_after is not None and end_date <= start_after:
                continue
            if end_at is not None and end_date > end_at:
                continue
            sequences.append(values[end_position - lookback + 1 : end_position + 1])
            targets.append(float(transform_target([target_values[end_position]])[0]))
            metadata.append({TICKER_COLUMN: ticker, DATE_COLUMN: end_date})

    if not sequences:
        return (
            np.empty((0, lookback, len(preprocessor.output_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(columns=[TICKER_COLUMN, DATE_COLUMN]),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def make_latest_sequence(
    frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    *,
    lookback: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build one target-free, ticker-isolated sequence for live LSTM inference."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    required = {TICKER_COLUMN, DATE_COLUMN, *preprocessor.feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing LSTM inference columns: {', '.join(missing)}")

    work = frame.sort_values([TICKER_COLUMN, DATE_COLUMN], kind="stable").copy()
    tickers = work[TICKER_COLUMN].astype(str).unique()
    if len(tickers) != 1:
        raise ValueError("live LSTM inference requires exactly one ticker")
    if len(work) < lookback:
        raise ValueError(f"LSTM requires at least {lookback} complete feature rows")

    latest_window = work.tail(lookback)
    transformed = preprocessor.transform(latest_window).to_numpy(dtype=np.float32)
    metadata = latest_window.loc[:, [TICKER_COLUMN, DATE_COLUMN]].tail(1).reset_index(drop=True)
    return transformed[np.newaxis, :, :], metadata
