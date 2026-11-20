#!/bin/sh
# Build the data on first start, then run the dashboard (default) or the API.
set -e
if [ ! -f data/processed/test_documents.jsonl ]; then
  echo "First start: downloading GovReport and building section trees..."
  uv run --no-sync python scripts/prepare_data.py
fi
PORT="${PORT:-}"
case "$1" in
  api) exec uv run --no-sync uvicorn structuralrag.api.routes:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  app) exec uv run --no-sync streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true ;;
  *) exec "$@" ;;
esac
