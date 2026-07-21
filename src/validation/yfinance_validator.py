"""Validation rules for normalized yfinance collection output."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.validation.result import ValidationResult

logger = logging.getLogger(__name__)


class YFinanceDataValidator:
    """Validate normalized yfinance metadata and daily prices."""

    REQUIRED_PRICE_COLUMNS = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
        "source",
    ]
    CORE_OHLC_COLUMNS = ["open", "high", "low", "close"]
    OPTIONAL_METADATA_TYPES = {
        "exchange": str,
        "currency": str,
        "sector": str,
        "industry": str,
        "country": str,
        "website": str,
        "market_cap": (int, float),
        "quote_type": str,
        "timezone": str,
    }

    def __init__(self, large_movement_threshold: float = 0.50) -> None:
        """Create a validator with a close-to-close movement warning threshold."""

        if large_movement_threshold <= 0:
            raise ValueError("large_movement_threshold must be greater than zero")
        self.large_movement_threshold = large_movement_threshold

    def validate_daily_prices(self, df: pd.DataFrame) -> ValidationResult:
        """Validate normalized daily-price rows without mutating the input frame."""

        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(df, pd.DataFrame):
            errors.append("Input must be a pandas DataFrame.")
            return self._result(errors=errors, warnings=warnings, row_count=0)

        row_count = len(df)
        work = df.copy(deep=True)
        logger.info("Validating %s yfinance daily price rows", row_count)

        missing_columns = [
            column for column in self.REQUIRED_PRICE_COLUMNS if column not in work.columns
        ]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}.")
            return self._result(errors=errors, warnings=warnings, row_count=row_count)

        if work.empty:
            errors.append("DataFrame is empty.")
            return self._result(errors=errors, warnings=warnings, row_count=row_count)

        duplicate_full_rows = int(work.duplicated().sum())
        if duplicate_full_rows:
            warnings.append(f"Duplicate full rows found: {duplicate_full_rows}.")

        self._validate_ticker(work, errors)
        parsed_dates = self._validate_dates(work, errors, warnings)
        numeric_ohlc = self._validate_ohlc(work, errors)
        self._validate_volume(work, errors, warnings)
        self._validate_optional_price_columns(work, warnings)
        self._validate_source(work, errors)

        if not numeric_ohlc.empty:
            self._validate_ohlc_relationships(numeric_ohlc, errors)
            self._validate_negative_prices(work, numeric_ohlc, errors)
            self._warn_large_movements(work, numeric_ohlc["close"], parsed_dates, warnings)

        duplicate_ticker_dates = int(work.duplicated(subset=["ticker", "date"]).sum())
        if duplicate_ticker_dates:
            errors.append(f"Duplicate ticker/date pairs found: {duplicate_ticker_dates}.")

        logger.info(
            "Daily price validation complete rows=%s errors=%s warnings=%s",
            row_count,
            len(errors),
            len(warnings),
        )
        return self._result(errors=errors, warnings=warnings, row_count=row_count)

    def validate_company_metadata(self, metadata: dict[str, Any]) -> ValidationResult:
        """Validate normalized yfinance company metadata."""

        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(metadata, dict):
            errors.append("Input must be a dictionary.")
            return self._result(errors=errors, warnings=warnings, row_count=0)

        logger.info("Validating yfinance metadata for ticker=%s", metadata.get("ticker"))

        if self._is_missing_text(metadata.get("ticker")):
            errors.append("ticker is missing or empty.")
        if self._is_missing_text(metadata.get("company_name")):
            errors.append("company_name is missing or empty.")
        if metadata.get("source") != "yfinance":
            errors.append('source is missing or not equal to "yfinance".')

        for field in ("exchange", "currency", "sector", "industry", "market_cap"):
            if metadata.get(field) is None:
                warnings.append(f"{field} is missing.")

        for field, expected_type in self.OPTIONAL_METADATA_TYPES.items():
            value = metadata.get(field)
            if value is not None and not isinstance(value, expected_type):
                warnings.append(f"{field} has unexpected type {type(value).__name__}.")

        logger.info(
            "Metadata validation complete errors=%s warnings=%s",
            len(errors),
            len(warnings),
        )
        return self._result(errors=errors, warnings=warnings, row_count=1)

    def _validate_ticker(self, df: pd.DataFrame, errors: list[str]) -> None:
        ticker_values = df["ticker"]
        invalid_tickers = ticker_values.isna() | ticker_values.astype("string").str.strip().eq("")
        if bool(invalid_tickers.any()):
            errors.append("ticker contains null or empty values.")

    def _validate_dates(
        self,
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
    ) -> pd.Series:
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        if bool(dates.isna().any()):
            errors.append("date contains null or invalid values.")

        today = pd.Timestamp.today().normalize()
        valid_dates = dates.dropna().dt.normalize()
        if bool((valid_dates > today).any()):
            errors.append("dates in the future found.")

        if not dates.is_monotonic_increasing:
            warnings.append("dates are not sorted ascending.")

        return dates

    def _validate_ohlc(self, df: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
        numeric_ohlc = pd.DataFrame(index=df.index)
        non_numeric_columns: list[str] = []
        null_columns: list[str] = []

        for column in self.CORE_OHLC_COLUMNS:
            original = df[column]
            numeric = pd.to_numeric(original, errors="coerce")
            numeric_ohlc[column] = numeric

            original_nulls = original.isna()
            non_numeric_values = numeric.isna() & ~original_nulls
            if bool(non_numeric_values.any()):
                non_numeric_columns.append(column)
            if bool(original_nulls.any()):
                null_columns.append(column)

        if non_numeric_columns:
            errors.append(f"Core OHLC fields contain non-numeric values: {non_numeric_columns}.")
        if null_columns:
            errors.append(f"Core OHLC fields contain null values: {null_columns}.")

        return numeric_ohlc

    def _validate_volume(
        self,
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        if bool((volume < 0).fillna(False).any()):
            errors.append("negative volume found.")
        if bool((volume == 0).fillna(False).any()):
            warnings.append("volume is zero.")

    def _validate_optional_price_columns(self, df: pd.DataFrame, warnings: list[str]) -> None:
        for column in ("adjusted_close", "dividends", "stock_splits"):
            if bool(df[column].isna().any()):
                warnings.append(f"{column} contains null values.")

    def _validate_source(self, df: pd.DataFrame, errors: list[str]) -> None:
        source = df["source"]
        if bool(source.isna().any()) or bool((source != "yfinance").any()):
            errors.append('source is missing or not equal to "yfinance".')

    def _validate_ohlc_relationships(self, prices: pd.DataFrame, errors: list[str]) -> None:
        if bool((prices["high"] < prices["low"]).fillna(False).any()):
            errors.append("high is less than low.")
        if bool((prices["high"] < prices["open"]).fillna(False).any()):
            errors.append("high is less than open.")
        if bool((prices["high"] < prices["close"]).fillna(False).any()):
            errors.append("high is less than close.")
        if bool((prices["low"] > prices["open"]).fillna(False).any()):
            errors.append("low is greater than open.")
        if bool((prices["low"] > prices["close"]).fillna(False).any()):
            errors.append("low is greater than close.")

    def _validate_negative_prices(
        self,
        df: pd.DataFrame,
        numeric_ohlc: pd.DataFrame,
        errors: list[str],
    ) -> None:
        price_columns = numeric_ohlc.copy()
        if "adjusted_close" in df.columns:
            price_columns["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")

        if bool((price_columns < 0).fillna(False).any().any()):
            errors.append("negative prices found.")

    def _warn_large_movements(
        self,
        df: pd.DataFrame,
        close_prices: pd.Series,
        dates: pd.Series,
        warnings: list[str],
    ) -> None:
        movement_frame = pd.DataFrame(
            {
                "ticker": df["ticker"],
                "date": dates,
                "close": close_prices,
            }
        ).dropna(subset=["ticker", "date", "close"])
        if movement_frame.empty:
            return

        movement_frame = movement_frame.sort_values(["ticker", "date"])
        close_change = movement_frame.groupby("ticker")["close"].pct_change().abs()
        if bool((close_change > self.large_movement_threshold).fillna(False).any()):
            warnings.append(
                "unusually large one-day close-to-close movement exceeds "
                f"{self.large_movement_threshold:.0%}."
            )

    @staticmethod
    def _is_missing_text(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == "")

    @staticmethod
    def _result(
        errors: list[str],
        warnings: list[str],
        row_count: int,
    ) -> ValidationResult:
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            row_count=row_count,
        )
