.PHONY: install data eval test lint api app all

install:
	uv sync

data:
	uv run python scripts/prepare_data.py

eval:
	uv run python scripts/run_evaluation.py

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests app
	uv run ruff format --check src scripts tests app

api:
	uv run uvicorn structuralrag.api.routes:app --reload

app:
	uv run streamlit run app/streamlit_app.py

all: install data eval test
