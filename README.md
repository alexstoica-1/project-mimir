# MIMIR

### Market Intelligence & Machine-learning for Investment Research

MIMIR is an end-to-end financial intelligence platform designed to collect, process, analyze, and serve financial market data through modern machine learning pipelines and APIs.

The platform aims to transform raw financial data into actionable insights by combining automated data collection, robust data engineering, predictive machine learning models, and a production-ready backend architecture.

This project is built as a production-oriented portfolio project with a strong emphasis on software engineering, reproducible machine learning workflows, and scalable system design.

---

## Vision

MIMIR aims to become a modular financial intelligence platform capable of:

- Collecting financial and market data from external APIs
- Storing and managing historical financial datasets
- Building reproducible feature engineering pipelines
- Training and evaluating machine learning models
- Serving predictions through a FastAPI backend
- Deploying the complete system with Docker
- Providing an extensible architecture for future AI-powered financial research tools

---

## Planned Architecture

```
                External APIs
                      │
                      ▼
              Data Collection
                      │
                      ▼
                PostgreSQL
                      │
                      ▼
           Data Validation & Cleaning
                      │
                      ▼
           Feature Engineering Pipeline
                      │
                      ▼
              Machine Learning
                      │
                      ▼
             Prediction Service
                      │
                      ▼
                  FastAPI
                      │
                      ▼
                 REST API
```

---

## Project Structure

```
project-mimir/
│
├── src/
│   ├── api/
│   ├── collectors/
│   ├── database/
│   ├── features/
│   ├── ml/
│   └── utils/
│
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│
├── notebooks/
├── models/
├── scripts/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Current Status

🚧 Project under active development.

The initial milestone focuses on building a complete end-to-end machine learning pipeline, including data ingestion, storage, preprocessing, model training, API deployment, and containerization.

## Volatility Forecasting

The processed feature dataset is expected at
`data/processed/v1/features_all_tickers.csv`. The forecasting workflow uses
chronological 70/15/15 train/validation/test splits and tracks runs in MLflow.

Run the full model comparison with:

```bash
/opt/miniconda3/envs/finance-ml-api/bin/python -m scripts.train_model \
  --data data/processed/v1/features_all_tickers.csv \
  --output-dir models/volatility \
  --experiment mimir-volatility \
  --tracking-uri sqlite:///mlflow.db
```

For a quick smoke test, use a small ticker subset and one LSTM epoch:

```bash
/opt/miniconda3/envs/finance-ml-api/bin/python -m scripts.train_model \
  --tickers AAPL MSFT \
  --lstm-epochs 1 \
  --no-local-lightgbm \
  --output-dir /tmp/mimir-volatility-smoke \
  --tracking-uri sqlite:////tmp/mimir-mlflow-smoke.db
```

Start the local MLflow UI in another terminal:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

The workflow trains per-ticker Student-t GARCH(1,1), pooled LightGBM, pooled
PyTorch LSTM, and optional per-ticker LightGBM models. It logs predictions,
metrics, feature importance, preprocessing artifacts, and model versions.

## Serving the selected model

The deployed model is the global LightGBM artifact at
`models/volatility/lightgbm_global.joblib`. The API reads the latest persisted
OHLCV data for a trained ticker together with SPY and VIX, builds the same
causal features used for training, and forecasts annualized realized volatility
for the next 20 trading days.

Start the local stack after the artifact has been trained:

```bash
docker compose up --build
```

The model directory is mounted read-only into the API container and stays out
of Git. Docker initializes an empty PostgreSQL schema on first start; ingest
market data for a supported ticker plus `SPY` and `^VIX` before requesting a
forecast. The Compose database is published on host port `5433` to avoid
clashing with the local PostgreSQL instance normally used during development.

Once data is available, use:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/model
curl -X POST http://127.0.0.1:8000/v1/predictions/AAPL
```

The API returns `404` for a ticker without persisted prices, and `422` when
the ticker was not part of model training, has too little history, or lacks
same-date SPY/VIX market context.

After the final evaluation, record the chosen model in local MLflow:

```bash
/opt/miniconda3/envs/finance-ml-api/bin/python -m scripts.promote_model \
  --model-name mimir-lightgbm-global --version 1 --alias champion
```
