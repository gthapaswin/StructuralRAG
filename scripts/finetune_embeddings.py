"""Fine-tune the embedding model on train-split questions with tree-based hard negatives.

Loss: CachedMultipleNegativesRankingLoss (GradCache: a batch of 32 in chunks of 8 to fit
laptop memory; in-batch negatives from other reports + hard negatives
from the same report, mined with the base model). Writes the model to finetune.embedding.output.

uv run python scripts/finetune_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from datasets import Dataset

from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import DenseRetriever, Encoder, query_prefix
from structuralrag.training.data import MODES, training_queries
from structuralrag.training.hard_negatives import mine_hard_negatives
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("finetune_embeddings")


def main() -> None:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", choices=MODES, default="ict", help="ict, or ict + LLM paraphrases")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ft, emb_cfg = cfg["finetune"], cfg["finetune"]["embedding"]
    base_name = cfg["retrieval"]["embedding_model"]
    out_dir = PROJECT_ROOT / emb_cfg["output" if args.questions == "ict" else "output_mixed"]

    corpus = SectionCorpus(load_documents(DATA_DIR / "train_documents.jsonl"))
    queries = training_queries(args.questions, cfg["seed"], llm_repeat=ft["llm_question_repeat"])
    t0 = time.perf_counter()
    base = DenseRetriever(corpus, Encoder(base_name, CACHE_DIR), View.TEXT)
    negs = mine_hard_negatives(corpus, queries, base.score_batch([q.question for q in queries]), ft["hard_negatives"])
    log.info("Mined hard negatives in %.0fs", time.perf_counter() - t0)

    n_neg = ft["hard_negatives"]
    prefix = query_prefix(base_name)
    rows = {"anchor": [], "positive": [], **{f"negative_{i}": [] for i in range(n_neg)}}
    for q, ns in zip(queries, negs, strict=True):
        if len(ns) < n_neg:
            continue  # every row needs the same number of columns
        gold = corpus.sections[corpus.uid_to_idx[next(iter(q.target_uids))]]
        rows["anchor"].append(prefix + q.question)
        rows["positive"].append(gold.text)
        for i, j in enumerate(ns):
            rows[f"negative_{i}"].append(corpus.sections[j].text)
    train = Dataset.from_dict(rows).shuffle(seed=cfg["seed"])
    log.info("Training rows: %d (of %d queries)", len(train), len(queries))

    # Eager attention: on Apple MPS the fused SDPA kernel has no dropout support during training.
    model = SentenceTransformer(base_name, model_kwargs={"attn_implementation": "eager"})
    model.max_seq_length = emb_cfg["max_seq_length"]
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "embedding"),
        num_train_epochs=emb_cfg["epochs"],
        per_device_train_batch_size=emb_cfg["batch_size"],
        learning_rate=emb_cfg["learning_rate"],
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # a duplicate text in a batch would be a false negative
        logging_steps=25,
        save_strategy="no",
        seed=cfg["seed"],
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train, loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    # Trained on 256-token passages to fit memory; used like the base model (512) at inference,
    # so base and fine-tuned encoders see exactly the same text.
    model.max_seq_length = 512
    model.save(str(out_dir))
    meta = {
        "base_model": base_name,
        "questions": args.questions,
        "train_rows": len(train),
        "train_seconds": train_s,
        **emb_cfg,
    }
    (out_dir / "training_info.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
