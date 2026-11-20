from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from structuralrag.utils.io import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "indexes"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_CONFIG).open() as f:
        cfg = yaml.safe_load(f)
    # Let deployments (e.g. Docker) point at Ollama elsewhere without editing the config.
    if os.environ.get("OLLAMA_HOST") and "llm" in cfg:
        cfg["llm"]["host"] = os.environ["OLLAMA_HOST"]
    return cfg
