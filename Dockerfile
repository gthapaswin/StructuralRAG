# StructuralRAG dashboard + API. CPU-only PyTorch (see pyproject.toml).
#   docker build -t structuralrag .
#   docker run -p 8501:8501 structuralrag              # dashboard
#   docker run -p 8000:8000 structuralrag api          # FastAPI
# First start downloads GovReport and builds the data (~2 min); mount a volume on
# /app/data to keep it:  -v structuralrag-data:/app/data
# LLM features (generative retrieval, cited answers) need Ollama: pass
# -e OLLAMA_HOST=http://host.docker.internal:11434 ; without it they are disabled.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY app ./app
COPY configs ./configs
COPY reports ./reports
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN uv sync --frozen --no-dev && chmod +x docker/entrypoint.sh

EXPOSE 8501 8000
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["app"]
