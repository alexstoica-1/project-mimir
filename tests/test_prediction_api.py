"""Deterministic unit and API tests for serving the global LightGBM artifact."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.dependencies import get_prediction_service
from src.api.main import create_app
from src.features.pipeline import BASE_PRICE_COLUMNS, FEATURE_COLUMNS, MarketFeaturePipeline
from src.ml.lightgbm_model import LightGBMRegressorModel
from src.services.prediction_service import (
    InsufficientHistoryError,
    LightGBMVolatilityPredictionService,
    MARKET_HISTORY_RANGES,
    MissingMarketContextError,
    build_model_metadata,
)


def _synthetic_prices(ticker: str, periods: int = 130) -> pd.DataFrame:
    """Create non-constant, date-aligned raw prices for inference tests."""

    dates = pd.bdate_range("2024-01-02", periods=periods)
    steps = np.arange(periods, dtype=float)
    if ticker == "^VIX":
        close = 20.0 + 0.03 * steps + 0.4 * np.sin(steps / 4.0)
    else:
        offset = 0.0002 if ticker == "SPY" else 0.0
        close = np.exp(np.log(100.0) + (0.001 + offset) * steps + 0.002 * np.sin(steps))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": close * 0.997,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000 + (steps * 1003).astype(int),
            "adjusted_close": close,
            "source": "yfinance",
        }
    )


class FakeMarketRepository:
    """In-memory implementation of the service's database read contract."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def company_exists(self, ticker: str) -> bool:
        return ticker in self.frames and ticker not in {"SPY", "^VIX"}

    def latest_price_date(self, ticker: str, source: str = "yfinance") -> date | None:
        frame = self.frames.get(ticker, pd.DataFrame())
        if frame.empty:
            return None
        return pd.Timestamp(frame["date"].max()).date()

    def get_price_history_frame(
        self,
        ticker: str,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        frame = self.frames.get(ticker, pd.DataFrame(columns=BASE_PRICE_COLUMNS)).copy()
        if start is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
        return frame


@pytest.fixture()
def trained_artifact(tmp_path):
    """Train a tiny valid global artifact with AAPL in its ticker encoder."""

    dates = pd.bdate_range("2022-01-03", periods=100)
    rows: list[dict[str, object]] = []
    for index, row_date in enumerate(dates):
        row = {
            "ticker": "AAPL",
            "date": row_date,
            "target_rv_20d": 0.18 + (index % 7) * 0.001,
        }
        for column_number, column in enumerate(FEATURE_COLUMNS):
            row[column] = 0.01 * (index + column_number + 1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    model = LightGBMRegressorModel().fit(
        frame.iloc[:70],
        frame.iloc[70:85],
        params={"n_estimators": 8, "num_leaves": 7, "min_child_samples": 1, "verbosity": -1},
    )
    path = tmp_path / "lightgbm_global.joblib"
    model.save(str(path))
    return path, model


@pytest.fixture()
def prediction_service(trained_artifact):
    """Create a service backed by fully aligned synthetic market data."""

    path, model = trained_artifact
    repository = FakeMarketRepository(
        {
            "AAPL": _synthetic_prices("AAPL"),
            "SPY": _synthetic_prices("SPY"),
            "^VIX": _synthetic_prices("^VIX"),
            "MSFT": _synthetic_prices("MSFT"),
        }
    )
    return LightGBMVolatilityPredictionService(
        model=model,
        metadata=build_model_metadata(model, path),
        repository=repository,
    )


@pytest.fixture()
def market_history_service(trained_artifact):
    """Create a service with enough complete rows for all history ranges."""

    path, model = trained_artifact
    repository = FakeMarketRepository(
        {
            "AAPL": _synthetic_prices("AAPL", periods=400),
            "SPY": _synthetic_prices("SPY", periods=400),
            "^VIX": _synthetic_prices("^VIX", periods=400),
            "MSFT": _synthetic_prices("MSFT", periods=400),
        }
    )
    return LightGBMVolatilityPredictionService(
        model=model,
        metadata=build_model_metadata(model, path),
        repository=repository,
    )


def test_prediction_service_uses_latest_targetless_feature_row(prediction_service) -> None:
    forecast = prediction_service.predict_latest("aapl")
    feature_row = MarketFeaturePipeline(prediction_service.repository).build_for_ticker(
        "AAPL",
        inference_ready=True,
    ).iloc[[-1]]
    direct_prediction = float(prediction_service.model.predict(feature_row)[0])

    assert forecast.ticker == "AAPL"
    assert forecast.as_of_date == date(2024, 7, 1)
    assert np.isfinite(forecast.predicted_rv_20d)
    assert forecast.predicted_rv_20d >= 0
    assert forecast.predicted_rv_20d == pytest.approx(direct_prediction)
    assert forecast.model.feature_count == len(FEATURE_COLUMNS)


def test_prediction_service_rejects_short_history(trained_artifact) -> None:
    path, model = trained_artifact
    repository = FakeMarketRepository(
        {
            "AAPL": _synthetic_prices("AAPL", periods=30),
            "SPY": _synthetic_prices("SPY", periods=30),
            "^VIX": _synthetic_prices("^VIX", periods=30),
        }
    )
    service = LightGBMVolatilityPredictionService(
        model=model,
        metadata=build_model_metadata(model, path),
        repository=repository,
    )

    with pytest.raises(InsufficientHistoryError):
        service.predict_latest("AAPL")


def test_prediction_service_requires_latest_market_context(prediction_service) -> None:
    prediction_service.repository.frames["^VIX"] = prediction_service.repository.frames["^VIX"].iloc[:-1]

    with pytest.raises(MissingMarketContextError):
        prediction_service.predict_latest("AAPL")


def test_market_summary_uses_the_latest_causal_feature_row(prediction_service) -> None:
    summary = prediction_service.get_market_summary("aapl")
    feature_row = MarketFeaturePipeline(prediction_service.repository).build_for_ticker(
        "AAPL",
        inference_ready=True,
    ).iloc[-1]

    assert summary.ticker == "AAPL"
    assert summary.as_of_date == pd.Timestamp(feature_row.date).date()
    assert summary.adjusted_close == pytest.approx(feature_row.adjusted_close)
    assert summary.return_1d == pytest.approx(np.expm1(feature_row.log_return))
    assert summary.return_5d == pytest.approx(feature_row.return_5d)
    assert summary.return_20d == pytest.approx(feature_row.return_20d)
    assert summary.rv_20d == pytest.approx(feature_row.rv_20d)
    assert summary.drawdown == pytest.approx(feature_row.drawdown)
    assert summary.volume_ratio_20d == pytest.approx(feature_row.volume_ratio_20d)
    assert not hasattr(summary, "target_rv_20d")


@pytest.mark.parametrize("history_range, expected_rows", MARKET_HISTORY_RANGES.items())
def test_market_history_uses_complete_causal_rows(
    market_history_service,
    history_range,
    expected_rows,
) -> None:
    history = market_history_service.get_market_history("AAPL", history_range)

    assert history.ticker == "AAPL"
    assert history.history_range == history_range
    assert len(history.points) == expected_rows
    assert history.available_observations >= expected_rows
    assert [point.date for point in history.points] == sorted(point.date for point in history.points)
    assert all(point.rv_20d >= 0 for point in history.points)
    assert all(not hasattr(point, "target_rv_20d") for point in history.points)


def test_market_history_rejects_incomplete_requested_range(prediction_service) -> None:
    with pytest.raises(InsufficientHistoryError, match="1y requires 252"):
        prediction_service.get_market_history("AAPL", "1y")


def test_api_health_model_and_prediction_routes(trained_artifact, prediction_service) -> None:
    path, _model = trained_artifact
    engine = create_engine("sqlite+pysqlite:///:memory:")
    test_session_factory = sessionmaker(bind=engine)
    app = create_app(model_path=path, session_factory=test_session_factory)
    app.dependency_overrides[get_prediction_service] = lambda: prediction_service

    with TestClient(app) as client:
        health = client.get("/health")
        model = client.get("/v1/model")
        prediction = client.post("/v1/predictions/AAPL")
        unsupported = client.post("/v1/predictions/MSFT")
        missing = client.post("/v1/predictions/ZZZ")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert model.status_code == 200
    assert model.json()["supported_tickers"] == ["AAPL"]
    assert prediction.status_code == 200
    assert prediction.json()["ticker"] == "AAPL"
    assert prediction.json()["predicted_rv_20d"] >= 0
    assert unsupported.status_code == 422
    assert missing.status_code == 404


def test_api_returns_market_history_and_expected_errors(trained_artifact, market_history_service) -> None:
    path, _model = trained_artifact
    engine = create_engine("sqlite+pysqlite:///:memory:")
    app = create_app(model_path=path, session_factory=sessionmaker(bind=engine))
    app.dependency_overrides[get_prediction_service] = lambda: market_history_service

    with TestClient(app) as client:
        history = client.get("/v1/market-data/AAPL?range=3m")
        unsupported = client.get("/v1/market-data/MSFT?range=3m")
        missing = client.get("/v1/market-data/ZZZ?range=3m")
        invalid_range = client.get("/v1/market-data/AAPL?range=2y")

    assert history.status_code == 200
    response = history.json()
    assert response["ticker"] == "AAPL"
    assert response["range"] == "3m"
    assert len(response["points"]) == 63
    assert set(response["points"][0]) == {
        "date", "adjusted_close", "rv_20d", "return_20d", "drawdown", "volume_ratio_20d",
    }
    assert unsupported.status_code == 422
    assert missing.status_code == 404
    assert invalid_range.status_code == 422


def test_api_returns_market_summary_and_expected_errors(trained_artifact, prediction_service) -> None:
    path, _model = trained_artifact
    engine = create_engine("sqlite+pysqlite:///:memory:")
    app = create_app(model_path=path, session_factory=sessionmaker(bind=engine))
    app.dependency_overrides[get_prediction_service] = lambda: prediction_service

    with TestClient(app) as client:
        summary = client.get("/v1/market-summary/AAPL")
        unsupported = client.get("/v1/market-summary/MSFT")
        missing = client.get("/v1/market-summary/ZZZ")

    assert summary.status_code == 200
    response = summary.json()
    assert set(response) == {
        "ticker",
        "as_of_date",
        "adjusted_close",
        "return_1d",
        "return_5d",
        "return_20d",
        "rv_20d",
        "drawdown",
        "volume_ratio_20d",
    }
    assert response["ticker"] == "AAPL"
    assert response["rv_20d"] >= 0
    assert response["volume_ratio_20d"] >= 0
    assert "target_rv_20d" not in response
    assert unsupported.status_code == 422
    assert missing.status_code == 404


def test_api_market_summary_requires_latest_market_context(trained_artifact, prediction_service) -> None:
    path, _model = trained_artifact
    engine = create_engine("sqlite+pysqlite:///:memory:")
    app = create_app(model_path=path, session_factory=sessionmaker(bind=engine))
    app.dependency_overrides[get_prediction_service] = lambda: prediction_service
    prediction_service.repository.frames["^VIX"] = prediction_service.repository.frames["^VIX"].iloc[:-1]

    with TestClient(app) as client:
        response = client.get("/v1/market-summary/AAPL")

    assert response.status_code == 422
    assert "matching SPY/VIX context" in response.json()["detail"]


def test_api_market_summary_rejects_short_history(trained_artifact) -> None:
    path, model = trained_artifact
    repository = FakeMarketRepository(
        {
            "AAPL": _synthetic_prices("AAPL", periods=30),
            "SPY": _synthetic_prices("SPY", periods=30),
            "^VIX": _synthetic_prices("^VIX", periods=30),
        }
    )
    service = LightGBMVolatilityPredictionService(
        model=model,
        metadata=build_model_metadata(model, path),
        repository=repository,
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    app = create_app(model_path=path, session_factory=sessionmaker(bind=engine))
    app.dependency_overrides[get_prediction_service] = lambda: service

    with TestClient(app) as client:
        response = client.get("/v1/market-summary/AAPL")

    assert response.status_code == 422
    assert "needs at least" in response.json()["detail"]


def test_dashboard_and_static_assets_are_served(trained_artifact) -> None:
    """The browser dashboard is packaged with the FastAPI application."""

    path, _model = trained_artifact
    engine = create_engine("sqlite+pysqlite:///:memory:")
    app = create_app(model_path=path, session_factory=sessionmaker(bind=engine))

    with TestClient(app) as client:
        dashboard = client.get("/")
        stylesheet = client.get("/static/styles.css")
        script = client.get("/static/app.js")

    assert dashboard.status_code == 200
    assert "MIMIR · Volatility Forecast" in dashboard.text
    assert 'href="/static/styles.css"' in dashboard.text
    assert stylesheet.status_code == 200
    assert "--background: #09111f" in stylesheet.text
    assert script.status_code == 200
    assert 'fetch("/v1/model")' in script.text
    assert "/v1/predictions/${encodeURIComponent(ticker)}" in script.text
    assert "/v1/market-summary/${encodeURIComponent(ticker)}" in script.text
    assert "/v1/market-data/${encodeURIComponent(ticker)}?range=${historyState.range}" in script.text
