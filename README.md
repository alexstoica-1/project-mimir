# MIMIR

Market Intelligence & Machine-learning for Investment Research (MIMIR) is a
market-data and volatility-forecasting project. It collects daily market data,
stores it in PostgreSQL, builds causal volatility features, trains forecasting
models, and serves the selected model through FastAPI.

## Implemented architecture

```text
                    yfinance
                       |
                       v
       YFinanceCollector + YFinanceDataValidator
                       |
                       v
          MarketDataIngestionService
                       |
                       v
        MarketRepository / PostgreSQL
          |                    |
          |                    +--> companies, daily_prices, ingestion_runs
          v
    MarketFeaturePipeline
          |
          +--> model-ready CSV --> GARCH / LightGBM / LSTM training --> MLflow
          |                                                    |
          |                                                    v
          |                         lightgbm_global.joblib + lstm_global.pt
          |
          +--> inference-ready causal feature history --> FastAPI --> LightGBM + LSTM prediction JSON
```

There are two distinct flows:

1. **Training:** build a CSV with historical targets, split it chronologically,
   train and compare models, and save the chosen artifact.
2. **Serving:** load the saved global LightGBM champion and global LSTM once,
   read the latest PostgreSQL price history, build only causal features, and
   return both forecasts from the same market date.

The serving flow never requires `target_rv_20d`, because that target represents
future volatility and is unknown on the latest market date.

## Database schema

PostgreSQL stores normalized source data before feature engineering.

| Table | Purpose | Important fields |
|---|---|---|
| `companies` | Company metadata | `ticker` (primary key), company name, exchange, sector, industry, market cap, source, timestamps |
| `daily_prices` | One daily OHLCV observation | `id`, `ticker` (foreign key), `date`, `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, dividends, splits, source, timestamps |
| `ingestion_runs` | Audit record for each ingest operation | `id`, ticker, source, status, started/completed time, fetched/written rows, warnings, errors |

`daily_prices` has a unique constraint on `(ticker, date, source)`. Tickers are
normalized to uppercase. The forecast requires the requested stock, `SPY` for
market context, and `^VIX` for implied-volatility context.

## Feature and target schema

The training dataset is written to:

```text
data/processed/v1/features_all_tickers.csv
```

It has **42 columns**:

| Group | Columns |
|---|---|
| Identity and raw prices | `ticker`, `date`, `open`, `high`, `low`, `close`, `volume`, `adjusted_close`, `source` |
| Returns | `log_return`, `abs_log_return`, `log_return_squared`, `return_5d`, `return_20d`, `return_60d` |
| Realized volatility | `rv_5d`, `rv_10d`, `rv_20d`, `rv_60d`, `rv_20d_lag1`, `rv_20d_lag5`, `rv_20d_lag20`, `rv_change_1d`, `vol_of_vol_20d` |
| Price range and drawdown | `log_high_low`, `log_close_open`, `overnight_gap`, `drawdown`, `max_drawdown_20d`, `max_drawdown_60d` |
| Volume and return distribution | `volume_ratio_5d`, `volume_ratio_20d`, `volume_volatility_20d`, `rolling_skew_20d`, `rolling_kurtosis_20d` |
| Market context | `market_return_20d`, `market_rv_20d`, `market_corr_60d`, `vix`, `vix_change_5d`, `vix_minus_market_rv` |
| Training target | `target_rv_20d` |

`target_rv_20d` is annualized realized volatility calculated from the next 20
trading days. All input features use data available on or before their own date.
Rows without a complete lookback or a future target are dropped for training.

For serving, the API uses the same **32 engineered feature columns** plus
`ticker` and `date`, but deliberately excludes `target_rv_20d` so the newest
complete feature row can be predicted.

## Project flow

### 1. Ingest market data

The ingestion service fetches company metadata and daily prices from yfinance,
validates them, upserts them into PostgreSQL, and records the run outcome.

```bash
conda run -n finance-ml-api python -m scripts.ingest_market_data \
  --tickers AAPL SPY '^VIX' --period 10y
```

### 2. Build the training dataset

The feature pipeline reads persisted prices through `MarketRepository`, creates
stock features, merges exact-date SPY/VIX context, creates the forward target,
and writes a deterministic CSV.

```bash
conda run -n finance-ml-api python -m scripts.build_features \
  --tickers AAPL AMZN BA CAT GOOGL JNJ JPM KO META MSFT NVDA PG TSLA V WMT XOM
```

`SPY` and `^VIX` must already be present in PostgreSQL; they are context assets,
not prediction tickers.

### 3. Train and compare models

The model dataset is split by global calendar date: 70% training, 15%
validation, and 15% final test. Rows are never shuffled across time.

```bash
/opt/miniconda3/envs/finance-ml-api/bin/python -m scripts.train_model \
  --data data/processed/v1/features_all_tickers.csv \
  --output-dir models/volatility \
  --experiment mimir-volatility \
  --tracking-uri sqlite:///mlflow.db
```

The workflow evaluates a naive `rv_20d` baseline, per-ticker Student-t
GARCH(1,1), global LightGBM, optional local LightGBM models, and a global LSTM.
It logs parameters, metrics, predictions, and artifacts to MLflow.

Start the local MLflow interface with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

The selected deployed model is the global LightGBM artifact:

```text
models/volatility/lightgbm_global.joblib
```

Its MLflow registered-model name is `mimir-lightgbm-global`; version `1` is
currently assigned the `champion` alias.

## Serving the LightGBM and LSTM models

FastAPI loads the saved LightGBM and LSTM artifacts once at startup. For a request such as
`POST /v1/predictions/AAPL`, the application:

1. verifies that AAPL is one of the tickers used to train the model;
2. reads AAPL, SPY, and VIX history from PostgreSQL;
3. builds complete causal feature rows through the latest usable date;
4. gives the latest row to LightGBM and the latest 60 rows to the LSTM;
5. predicts the next 20-trading-day annualized realized volatility with both models; and
6. returns the ticker, feature date, champion identity, and both model versions.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms that the API can reach PostgreSQL and has loaded both models. |
| `GET /v1/model` | Returns the LightGBM champion and LSTM catalog, including version, target, horizon, and input requirements. |
| `POST /v1/predictions/{ticker}` | Returns the latest 20-trading-day volatility forecasts from both models. |
| `GET /v1/market-summary/{ticker}` | Returns the latest causal price, return, volatility, drawdown, and volume indicators. |
| `GET /v1/market-data/{ticker}?range=1y` | Returns causal engineered history for a trained ticker; ranges are `1m`, `3m`, `6m`, `1y`, or `5y` trading observations. |

Example response:

```json
{
  "ticker": "AAPL",
  "as_of_date": "2026-08-07",
  "target_name": "target_rv_20d",
  "unit": "annualized_decimal_volatility",
  "forecast_horizon_trading_days": 20,
  "champion_model_id": "lightgbm-global",
  "predictions": [
    {"model_id": "lightgbm-global", "display_name": "LightGBM", "model_version": "...", "predicted_rv_20d": 0.2656817},
    {"model_id": "lstm-global", "display_name": "LSTM", "model_version": "...", "predicted_rv_20d": 0.2619021}
  ]
}
```

The API returns `404` when no persisted price data exists for a ticker, `422`
when the ticker is unsupported, lacks sufficient history, or has no same-date
SPY/VIX context, and `503` when the model cannot serve a valid prediction.

### Browser dashboard

The API also serves a small dashboard at `GET /`. It loads the supported ticker
list from `/v1/model`, requests both forecasts through the same prediction endpoint,
and displays LightGBM as the champion alongside the LSTM comparison. Its model
details box lets you select either model to view its version, target, horizon,
feature count, and input requirement. It also
provides 3-month, 6-month, and 1-year market-history controls that show native
adjusted-close and `rv_20d` charts, a current market snapshot, and the latest
20 causal indicator rows.

When the Docker stack is running, open:

```text
http://127.0.0.1:8000/
```

## Docker deployment

Docker Compose runs PostgreSQL and the FastAPI service. The model directory is
mounted read-only into the API container; trained models and MLflow artifacts
are intentionally not committed to Git.

```bash
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`. The Compose PostgreSQL database
is published on host port `5433` to avoid a conflict with a local PostgreSQL
server on port `5432`.

The Docker database starts empty. To load data into it from the host, use its
database URL when running the ingest script:

```bash
DATABASE_URL='postgresql+psycopg2://mimir:mimir@localhost:5433/mimir' \
conda run -n finance-ml-api python -m scripts.ingest_market_data \
  --tickers AAPL SPY '^VIX' --period 10y
```

Then test the deployed service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/model
curl -X POST http://127.0.0.1:8000/v1/predictions/AAPL
```

## Development

Copy `.env.example` to `.env` and set local values as needed. `.env` is ignored
by Git; `.env.example` is the committed, non-secret configuration template.

Run the automated tests with:

```bash
conda run -n finance-ml-api python -m pytest -q
```
