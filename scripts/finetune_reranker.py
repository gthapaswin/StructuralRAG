"""Fine-tune a cross-encoder reranker on train-split questions.

Each question gives one positive (its gold section) and `hard_negatives` negatives: the
sections of the same report that the base dense model ranks highest, skipping sections
that are ancestors or descendants of the gold one. Inputs are "hierarchy path + text",
so the reranker sees where each section sits in the report. Loss: binary cross-entropy.

uv run python scripts/finetune_reranker.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from datasets import Dataset

from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import DenseRetriever, Encoder
from structuralrag.retrieval.reranker import passage
from structuralrag.training.data import MODES, training_queries
from structuralrag.training.hard_negatives import mine_hard_negatives
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("finetune_reranker")


def main() -> None:
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", choices=MODES, default="ict", help="ict, or ict + LLM paraphrases")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ft, rr = cfg["finetune"], cfg["finetune"]["reranker"]
    out_dir = PROJECT_ROOT / rr["output" if args.questions == "ict" else "output_mixed"]

    corpus = SectionCorpus(load_documents(DATA_DIR / "train_documents.jsonl"))
    queries = training_queries(
        args.questions, cfg["seed"], total=rr["train_queries"], llm_repeat=ft["llm_question_repeat"]
    )
    dense = DenseRetriever(corpus, Encoder(cfg["retrieval"]["embedding_model"], CACHE_DIR), View.TEXT)
    negs = mine_hard_negatives(corpus, queries, dense.score_batch([q.question for q in queries]), ft["hard_negatives"])

    rows: dict[str, list] = {"query": [], "passage": [], "label": []}
    for q, ns in zip(queries, negs, strict=True):
        gold = corpus.uid_to_idx[next(iter(q.target_uids))]
        for i, label in [(gold, 1.0)] + [(j, 0.0) for j in ns]:
            rows["query"].append(q.question)
            rows["passage"].append(passage(corpus.sections[i]))
            rows["label"].append(label)
    train = Dataset.from_dict(rows).shuffle(seed=cfg["seed"])
    n_pos = int(sum(rows["label"]))
    log.info("Training pairs: %d (%d positive)", len(train), n_pos)

    model = CrossEncoder(rr["base_model"], num_labels=1, max_length=rr["max_length"])
    targs = CrossEncoderTrainingArguments(
        output_dir=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "reranker"),
        num_train_epochs=rr["epochs"],
        per_device_train_batch_size=rr["batch_size"],
        learning_rate=rr["learning_rate"],
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        seed=cfg["seed"],
        report_to="none",
    )
    # Positives are 1 in (1 + hard_negatives): weight them so the loss is not dominated by negatives.
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(ft["hard_negatives"])))
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train, loss=loss)
    t0 = time.perf_counter()
    trainer.train()
    meta = {
        "base_model": rr["base_model"],
        "questions": args.questions,
        "train_queries": len(queries),
        "train_pairs": len(train),
        "train_seconds": time.perf_counter() - t0,
        **rr,
    }
    model.save_pretrained(str(out_dir))
    (out_dir / "training_info.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
