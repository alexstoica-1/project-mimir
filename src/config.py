from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the prediction API and database clients."""

    database_url: str = "postgresql+psycopg2://mimir:mimir@localhost:5432/mimir"
    model_path: Path = Path("models/volatility/lightgbm_global.joblib")
    lstm_model_path: Path = Path("models/volatility/lstm_global.pt")
    market_data_source: str = "yfinance"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()  # type: ignore
