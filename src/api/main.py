"""FastAPI application serving the selected global LightGBM volatility model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, status
from sqlalchemy import text

from src.api.routes import models, predictions
from src.config import settings
from src.database.connection import SessionLocal
from src.ml.predict import load_lightgbm
from src.schemas.prediction import HealthResponse
from src.services.prediction_service import ModelMetadata, build_model_metadata


def create_app(
    *,
    model_path: str | Path | None = None,
    session_factory: Callable[[], Any] = SessionLocal,
    market_data_source: str | None = None,
) -> FastAPI:
    """Create an app that loads one global LightGBM artifact at startup."""

    configured_model_path = Path(model_path or settings.model_path)
    configured_source = market_data_source or settings.market_data_source

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not configured_model_path.is_file():
            raise RuntimeError(f"Configured LightGBM artifact does not exist: {configured_model_path}")
        try:
            model = load_lightgbm(configured_model_path)
            metadata = build_model_metadata(model, configured_model_path)
            with session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            raise RuntimeError(f"Prediction API startup checks failed: {exc}") from exc

        app.state.prediction_model = model
        app.state.model_metadata = metadata
        app.state.session_factory = session_factory
        app.state.market_data_source = configured_source
        yield

    app = FastAPI(
        title="MIMIR Volatility Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(models.router)
    app.include_router(predictions.router)

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
        return HealthResponse(model_name=metadata.name, model_version=metadata.version)

    return app


app = create_app()
