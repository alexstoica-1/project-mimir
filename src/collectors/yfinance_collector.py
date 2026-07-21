"""yfinance-backed market data collection utilities."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceCollectionError(RuntimeError):
    """Raised when yfinance data cannot be collected or normalized."""


class YFinanceCollector:
    """Collect company metadata and daily prices from yfinance.

    Empty price responses are returned as an empty DataFrame with the expected
    schema. Malformed inputs, yfinance failures, and malformed non-empty
    responses raise YFinanceCollectionError.
    """

    PRICE_COLUMNS = [
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

    _CORE_YFINANCE_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def fetch_company_metadata(self, ticker: str) -> dict[str, Any]:
        """Fetch and normalize company metadata for one ticker."""

        normalized_ticker = self._normalize_ticker(ticker)
        logger.info("Requesting yfinance metadata for ticker=%s", normalized_ticker)

        try:
            ticker_obj = yf.Ticker(normalized_ticker)
            info = self._safe_mapping(ticker_obj.get_info())
            fast_info = self._safe_mapping(getattr(ticker_obj, "fast_info", None))
            history_metadata = self._safe_mapping(ticker_obj.get_history_metadata())
        except Exception as exc:
            logger.exception("yfinance metadata request failed for ticker=%s", normalized_ticker)
            raise YFinanceCollectionError(
                f"Failed to fetch yfinance metadata for {normalized_ticker}"
            ) from exc

        metadata = {
            "ticker": normalized_ticker,
            "company_name": self._first_present(info, "longName", "shortName"),
            "exchange": self._first_present(info, "exchange", "fullExchangeName"),
            "quote_type": self._first_present(info, "quoteType"),
            "currency": self._first_present(info, "currency") or fast_info.get("currency"),
            "timezone": (
                self._first_present(info, "timeZoneFullName", "timeZoneShortName")
                or history_metadata.get("timezone")
                or history_metadata.get("exchangeTimezoneName")
            ),
            "sector": self._first_present(info, "sector"),
            "industry": self._first_present(info, "industry"),
            "country": self._first_present(info, "country"),
            "website": self._first_present(info, "website"),
            "market_cap": self._first_present(info, "marketCap") or fast_info.get("market_cap"),
            "source": "yfinance",
        }

        missing_fields = [
            key for key, value in metadata.items() if key not in {"ticker", "source"} and value is None
        ]
        if missing_fields:
            logger.warning(
                "Missing optional yfinance metadata for ticker=%s fields=%s",
                normalized_ticker,
                missing_fields,
            )

        return metadata

    def fetch_daily_prices(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        period: str | None = "1y",
    ) -> pd.DataFrame:
        """Fetch and normalize daily historical price data for one ticker."""

        normalized_ticker = self._normalize_ticker(ticker)
        self._validate_date_range(start=start, end=end)

        request_kwargs: dict[str, Any] = {
            "tickers": normalized_ticker,
            "auto_adjust": False,
            "actions": True,
            "progress": False,
            "threads": False,
        }
        if start or end:
            request_kwargs["start"] = start
            request_kwargs["end"] = end
            logger.info(
                "Requesting yfinance daily prices for ticker=%s start=%s end=%s",
                normalized_ticker,
                start,
                end,
            )
            if period is not None:
                logger.info(
                    "Ignoring period=%s because explicit start/end date arguments were supplied",
                    period,
                )
        else:
            request_kwargs["period"] = period
            logger.info(
                "Requesting yfinance daily prices for ticker=%s period=%s",
                normalized_ticker,
                period,
            )

        try:
            raw_prices = yf.download(**request_kwargs)
        except Exception as exc:
            logger.exception("yfinance price request failed for ticker=%s", normalized_ticker)
            raise YFinanceCollectionError(
                f"Failed to fetch yfinance daily prices for {normalized_ticker}"
            ) from exc

        normalized_prices = self._normalize_daily_prices(raw_prices, normalized_ticker)
        self._log_price_summary(normalized_ticker, normalized_prices)
        return normalized_prices

    @classmethod
    def empty_price_frame(cls) -> pd.DataFrame:
        """Return an empty daily price DataFrame with the normalized schema."""

        return pd.DataFrame(columns=cls.PRICE_COLUMNS)

    @classmethod
    def _normalize_daily_prices(cls, raw_prices: Any, ticker: str) -> pd.DataFrame:
        if raw_prices is None:
            raise YFinanceCollectionError("yfinance returned None for daily prices")
        if not isinstance(raw_prices, pd.DataFrame):
            raise YFinanceCollectionError(
                f"Expected yfinance daily prices to be a DataFrame, got {type(raw_prices)!r}"
            )
        if raw_prices.empty:
            logger.warning("yfinance returned no daily price rows for ticker=%s", ticker)
            return cls.empty_price_frame()

        prices = raw_prices.copy()
        prices = cls._select_single_ticker_columns(prices, ticker)

        missing_core_columns = sorted(cls._CORE_YFINANCE_COLUMNS.difference(prices.columns))
        if missing_core_columns:
            raise YFinanceCollectionError(
                f"yfinance response missing required price columns: {missing_core_columns}"
            )

        if "Adj Close" not in prices.columns:
            logger.warning(
                "Missing optional Adj Close column for ticker=%s; adjusted_close will be null",
                ticker,
            )
            prices["Adj Close"] = pd.NA
        if "Dividends" not in prices.columns:
            logger.warning(
                "Missing optional Dividends column for ticker=%s; dividends will be null",
                ticker,
            )
            prices["Dividends"] = pd.NA
        if "Stock Splits" not in prices.columns:
            logger.warning(
                "Missing optional Stock Splits column for ticker=%s; stock_splits will be null",
                ticker,
            )
            prices["Stock Splits"] = pd.NA

        prices = prices.reset_index()
        date_column = cls._find_date_column(prices)
        prices = prices.rename(
            columns={
                date_column: "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            }
        )

        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        if getattr(prices["date"].dt, "tz", None) is not None:
            prices["date"] = prices["date"].dt.tz_convert(None)
        prices["date"] = prices["date"].dt.normalize()

        prices["ticker"] = ticker
        prices["source"] = "yfinance"
        prices = prices[cls.PRICE_COLUMNS]
        prices = prices.sort_values("date", ascending=True).drop_duplicates().reset_index(drop=True)
        return prices

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        if not isinstance(ticker, str):
            raise YFinanceCollectionError("ticker must be a non-empty string")

        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise YFinanceCollectionError("ticker must be a non-empty string")

        return normalized_ticker

    @classmethod
    def _validate_date_range(cls, start: str | None, end: str | None) -> None:
        start_date = cls._parse_iso_date(start, "start") if start is not None else None
        end_date = cls._parse_iso_date(end, "end") if end is not None else None

        if start_date is not None and end_date is not None and start_date >= end_date:
            raise YFinanceCollectionError("start must be earlier than end")

    @staticmethod
    def _parse_iso_date(value: str, field_name: str) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise YFinanceCollectionError(
                f"{field_name} must be a valid ISO date string in YYYY-MM-DD format"
            ) from exc

    @staticmethod
    def _safe_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _select_single_ticker_columns(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [str(column) for column in prices.columns]
            return prices

        for level in range(prices.columns.nlevels):
            level_values = {str(value).upper() for value in prices.columns.get_level_values(level)}
            if ticker in level_values:
                selected = prices.xs(ticker, axis=1, level=level, drop_level=True)
                selected.columns = [str(column) for column in selected.columns]
                return selected

        for level in range(prices.columns.nlevels):
            if len(prices.columns.get_level_values(level).unique()) == 1:
                selected = prices.droplevel(level, axis=1)
                selected.columns = [str(column) for column in selected.columns]
                return selected

        flattened = prices.copy()
        flattened.columns = [
            "_".join(str(part) for part in column if str(part))
            for column in flattened.columns.to_flat_index()
        ]
        return flattened

    @staticmethod
    def _find_date_column(prices: pd.DataFrame) -> str:
        for candidate in ("Date", "Datetime", "index"):
            if candidate in prices.columns:
                return candidate
        raise YFinanceCollectionError("Could not identify yfinance date index column")

    @staticmethod
    def _log_price_summary(ticker: str, prices: pd.DataFrame) -> None:
        row_count = len(prices)
        if prices.empty:
            logger.info("Returned 0 daily price rows for ticker=%s", ticker)
            return

        logger.info(
            "Returned %s daily price rows for ticker=%s earliest_date=%s latest_date=%s",
            row_count,
            ticker,
            prices["date"].min(),
            prices["date"].max(),
        )
