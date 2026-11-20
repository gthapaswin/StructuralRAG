"""Generate validated LLM-written questions (needs Ollama).

Test split: evaluation questions. Train split: paraphrased training questions for fine-tuning.

uv run python scripts/generate_questions.py [--split test|train] [--per-doc 2]
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
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--max-docs", type=int, default=None, help="use only the first N reports (time budget)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    llm = cfg["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    client.check()

    docs = load_documents(DATA_DIR / f"{args.split}_documents.jsonl")[: args.max_docs]
    t0 = time.perf_counter()
    queries, stats = generate_llm_questions(
        docs, client, split=args.split, per_doc=args.per_doc or llm["questions_per_document"], seed=cfg["seed"]
    )
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["model"] = llm["model"]
    suffix = "" if args.split == "test" else f"_{args.split}"
    out = DATA_DIR / f"{args.split}_queries_llm.jsonl"
    write_jsonl(out, queries)
    write_json(DATA_DIR / f"llm_question_stats{suffix}.json", stats)
    print(f"{len(queries)} questions -> {out}\n{stats}")
    for q in queries[:5]:
        print("-", q.question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
