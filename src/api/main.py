"""FastAPI application serving the selected global LightGBM volatility model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from src.api.routes import market_data, models, predictions
from src.config import settings
from src.database.connection import SessionLocal
from src.ml.predict import load_lightgbm, load_lstm
from src.schemas.prediction import HealthResponse
from src.services.prediction_service import (
    ModelMetadata,
    build_lstm_model_metadata,
    build_model_metadata,
    validate_served_model_compatibility,
)

STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"


def create_app(
    *,
    model_path: str | Path | None = None,
    lstm_model_path: str | Path | None = None,
    session_factory: Callable[[], Any] = SessionLocal,
    market_data_source: str | None = None,
) -> FastAPI:
    """Create an app that loads compatible LightGBM and LSTM artifacts at startup."""

    configured_model_path = Path(model_path or settings.model_path)
    configured_lstm_model_path = Path(lstm_model_path or settings.lstm_model_path)
    configured_source = market_data_source or settings.market_data_source

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        missing_artifacts = [
            str(path)
            for path in (configured_model_path, configured_lstm_model_path)
            if not path.is_file()
        ]
        if missing_artifacts:
            raise RuntimeError(f"Configured model artifact does not exist: {', '.join(missing_artifacts)}")
        try:
            model = load_lightgbm(configured_model_path)
            metadata = build_model_metadata(model, configured_model_path)
            lstm_model = load_lstm(configured_lstm_model_path)
            lstm_metadata = build_lstm_model_metadata(lstm_model, configured_lstm_model_path)
            validate_served_model_compatibility(model, lstm_model)
            with session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            raise RuntimeError(f"Prediction API startup checks failed: {exc}") from exc

        app.state.prediction_model = model
        app.state.model_metadata = metadata
        app.state.lstm_model = lstm_model
        app.state.lstm_model_metadata = lstm_metadata
        app.state.session_factory = session_factory
        app.state.market_data_source = configured_source
        yield

    app = FastAPI(
        title="MIMIR Volatility Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    app.include_router(models.router)
    app.include_router(predictions.router)
    app.include_router(market_data.router)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the small browser dashboard for the deployed model."""

        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health(request: Request) -> HealthResponse:
        """Confirm that the current process can reach its database and model."""

        try:
            with request.app.state.session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is unavailable",
            ) from exc
        metadata = request.app.state.model_metadata
        return HealthResponse(
            model_name=metadata.name,
            model_version=metadata.version,
            served_models=[metadata.name, request.app.state.lstm_model_metadata.name],
        )

    return app


app = create_app()
