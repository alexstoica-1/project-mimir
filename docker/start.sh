#!/usr/bin/env sh
set -eu

python -m scripts.create_database_tables
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
