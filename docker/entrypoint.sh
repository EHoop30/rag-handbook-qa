#!/usr/bin/env bash
# Ingest the corpus, then serve. The database is already healthy because
# docker-compose gates this container on the db healthcheck.
set -euo pipefail

echo "Ingesting corpus into pgvector ..."
python -m scripts.ingest

echo "Starting API on :8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
