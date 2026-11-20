"""Generate validated LLM-written questions for the test documents (needs Ollama).

uv run python scripts/generate_questions.py [--per-doc 2]
"""

from __future__ import annotations

import argparse
import logging
import time

from structuralrag.data_generation.llm_questions import generate_llm_questions
from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, load_config
from structuralrag.utils.io import load_documents, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-doc", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    llm = cfg["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    client.check()

    docs = load_documents(DATA_DIR / "test_documents.jsonl")
    t0 = time.perf_counter()
    queries, stats = generate_llm_questions(
        docs, client, split="test", per_doc=args.per_doc or llm["questions_per_document"], seed=cfg["seed"]
    )
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["model"] = llm["model"]
    write_jsonl(DATA_DIR / "test_queries_llm.jsonl", queries)
    write_json(DATA_DIR / "llm_question_stats.json", stats)
    print(f"{len(queries)} questions -> {DATA_DIR / 'test_queries_llm.jsonl'}\n{stats}")
    for q in queries[:5]:
        print("-", q.question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
