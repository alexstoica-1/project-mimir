"""FastAPI dependencies for database sessions and deployed model access."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.database.repository import MarketRepository
from src.services.prediction_service import LightGBMVolatilityPredictionService, ModelMetadata


def get_session(request: Request) -> Generator[Session, None, None]:
    """Create and close one database session for a request."""

    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_model_metadata(request: Request) -> ModelMetadata:
    """Return metadata for the artifact loaded during application startup."""

    return request.app.state.model_metadata


def get_served_models_metadata(request: Request) -> tuple[ModelMetadata, ModelMetadata]:
    """Return champion and comparison-model metadata loaded at startup."""

    return request.app.state.model_metadata, request.app.state.lstm_model_metadata


def get_prediction_service(
    request: Request,
    session: Session = Depends(get_session),
) -> LightGBMVolatilityPredictionService:
    """Create a request-scoped service using the cached global model."""

    return LightGBMVolatilityPredictionService(
        model=request.app.state.prediction_model,
        metadata=request.app.state.model_metadata,
        lstm_model=request.app.state.lstm_model,
        lstm_metadata=request.app.state.lstm_model_metadata,
        repository=MarketRepository(session),
        source=request.app.state.market_data_source,
    )
