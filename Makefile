.PHONY: install data eval test lint api app all nq-data heading-study heading-study-llm

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

# Heading study (when do section headings help?). Needs data/raw/nq/validation_{0,1}.parquet
# (see scripts/prepare_nq.py) and, for the LLM-headings step, Ollama with qwen3:8b.
nq-data:
	uv run python scripts/prepare_nq.py

heading-study: nq-data
	uv run python scripts/run_heading_study.py --datasets govreport_dev govreport qasper nq
	uv run python scripts/run_heading_study.py --datasets govreport qasper nq --variants real placebo_s0 generic_1.0 --encoder artifacts/models/bge-small-govreport-mixed --tag ft
	uv run python scripts/run_heading_study.py --datasets govreport qasper nq --variants real placebo_s0 generic_1.0 --encoder BAAI/bge-base-en-v1.5 --tag bgebase
	uv run python scripts/analyze_heading_study.py
	uv run python scripts/heading_examples.py

heading-study-llm:
	uv run python scripts/rewrite_headings.py
	uv run python scripts/run_heading_study.py --datasets govreport qasper nq --variants llm --merge
	uv run python scripts/analyze_heading_study.py
