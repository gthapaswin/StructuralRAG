"""LoRA fine-tune a small LLM for generative retrieval (the STAIR setting), with MLX.

Training example = the same prompt the zero-shot retriever uses (system prompt, table of
contents with snippets, question) -> {"sections": [gold, ...]}. The gold id comes first,
followed by the report's hardest negatives from the base dense model, so the model learns
to output a ranked list rather than a single guess. Train: GovReport train split.
Validation (for the loss curve only): dev ICT queries. Test reports are never used.

uv run --group finetune python scripts/finetune_generative.py [--prepare-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys

import yaml

from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import DenseRetriever, Encoder
from structuralrag.retrieval.generative import SYSTEM, render_toc, toc_prompt
from structuralrag.schema import Query
from structuralrag.training.hard_negatives import mine_hard_negatives
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("finetune_generative")
SFT_DIR = DATA_DIR / "sft"


def build_examples(
    split: str, n: int, max_ids: int, max_toc_words: int, model: str, seed: int
) -> tuple[list[dict], dict]:
    corpus = SectionCorpus(load_documents(DATA_DIR / f"{split}_documents.jsonl"))
    tocs = {d: render_toc(doc, snippets=True) for d, doc in corpus.docs.items()}
    queries: list[Query] = [
        q
        for q in load_queries(DATA_DIR / f"{split}_queries.jsonl")
        if len(tocs[q.document_id].split()) <= max_toc_words
    ]
    queries = random.Random(seed).sample(queries, min(n, len(queries)))
    dense = DenseRetriever(corpus, Encoder(model, CACHE_DIR), View.TEXT)
    negs = mine_hard_negatives(corpus, queries, dense.score_batch([q.question for q in queries]), max_ids - 1)
    rows = []
    for q, ns in zip(queries, negs, strict=True):
        ids = [q.target_section_ids[0]] + [corpus.sections[j].section_id for j in ns]
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": toc_prompt(tocs[q.document_id], q.question, max_ids)},
                    {"role": "assistant", "content": json.dumps({"sections": ids})},
                ]
            }
        )
    stats = {"examples": len(rows), "reports": len({q.document_id for q in queries})}
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    g, max_ids = cfg["finetune"]["llm"], cfg["llm"]["max_ids"]
    model = cfg["retrieval"]["embedding_model"]

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, split, n in (("train", "train", g["train_examples"]), ("valid", "dev", g["valid_examples"])):
        rows, stats[name] = build_examples(split, n, max_ids, g["max_toc_words"], model, cfg["seed"])
        with (SFT_DIR / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    (SFT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("SFT data: %s", stats)
    if args.prepare_only:
        return

    adapter = PROJECT_ROOT / g["output"]
    adapter.mkdir(parents=True, exist_ok=True)
    lora_cfg = {
        "model": g["base_model"],
        "train": True,
        "data": str(SFT_DIR),
        "fine_tune_type": "lora",
        "mask_prompt": True,  # loss only on the answer, not on the table of contents
        "num_layers": g["num_layers"],
        "batch_size": g["batch_size"],
        "iters": g["iters"],
        "learning_rate": g["learning_rate"],
        "max_seq_length": 4096,
        "grad_checkpoint": True,
        "steps_per_report": 25,
        "steps_per_eval": 200,
        "val_batches": 50,
        "save_every": 250,
        "adapter_path": str(adapter),
        "seed": cfg["seed"],
    }
    cfg_path = adapter / "lora_config.yaml"
    cfg_path.write_text(yaml.safe_dump(lora_cfg, sort_keys=False))
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg_path)], check=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
