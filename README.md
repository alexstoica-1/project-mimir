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