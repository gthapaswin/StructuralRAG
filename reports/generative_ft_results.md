# Fine-tuned generative retrieval

`Qwen/Qwen3-1.7B` with a LoRA adapter trained on GovReport train-split questions (MLX), vs the same model zero-shot. Prompt: table of contents with snippets. Document scope. No constrained decoding with MLX: invalid ids are dropped (see `valid_output`).


## ICT questions (n = 300)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | valid_output | mean_returned |
|---|---|---|---|---|---|---|
| structure-aware | 0.563 | 0.893 | 0.706 | 0.771 | 1.000 | 9.700 |
| generative 1.7B zero-shot | 0.207 | 0.323 | 0.257 | 0.274 | 0.947 | 2.067 |
| generative 1.7B fine-tuned | 0.473 | 0.677 | 0.549 | 0.581 | 1.000 | 4.800 |
| generative 1.7B fine-tuned + structure | 0.513 | 0.693 | 0.630 | 0.709 | 1.000 | 9.700 |
| generative qwen3:8b zero-shot (earlier run) | 0.380 | 0.723 | 0.510 | 0.563 | 1.000 | 4.420 |
| generative qwen3:8b + structure (earlier run) | 0.503 | 0.777 | 0.641 | 0.718 | 1.000 | 9.700 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| generative 1.7B fine-tuned vs generative 1.7B zero-shot | mrr@10 | 0.292 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs generative 1.7B zero-shot | recall@1 | 0.267 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs structure-aware | mrr@10 | -0.157 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs structure-aware | recall@1 | -0.090 | 0.007 | 0.017 |
| generative 1.7B fine-tuned + structure vs structure-aware | mrr@10 | -0.076 | 0.001 | 0.000 |
| generative 1.7B fine-tuned + structure vs structure-aware | recall@1 | -0.050 | 0.098 | 0.162 |


## LLM-written questions (n = 398)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | valid_output | mean_returned |
|---|---|---|---|---|---|---|
| structure-aware | 0.799 | 0.975 | 0.877 | 0.907 | 1.000 | 9.606 |
| generative 1.7B zero-shot | 0.327 | 0.465 | 0.388 | 0.408 | 0.842 | 2.050 |
| generative 1.7B fine-tuned | 0.638 | 0.802 | 0.704 | 0.729 | 1.000 | 4.832 |
| generative 1.7B fine-tuned + structure | 0.706 | 0.824 | 0.784 | 0.834 | 1.000 | 9.606 |
| generative qwen3:8b zero-shot (earlier run) | 0.641 | 0.864 | 0.732 | 0.766 | 1.000 | 4.304 |
| generative qwen3:8b + structure (earlier run) | 0.751 | 0.872 | 0.826 | 0.867 | 1.000 | 9.606 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| generative 1.7B fine-tuned vs generative 1.7B zero-shot | mrr@10 | 0.316 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs generative 1.7B zero-shot | recall@1 | 0.312 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs structure-aware | mrr@10 | -0.173 | 0.000 | 0.000 |
| generative 1.7B fine-tuned vs structure-aware | recall@1 | -0.161 | 0.000 | 0.000 |
| generative 1.7B fine-tuned + structure vs structure-aware | mrr@10 | -0.093 | 0.000 | 0.000 |
| generative 1.7B fine-tuned + structure vs structure-aware | recall@1 | -0.093 | 0.000 | 0.002 |


Uncached latency per query: {'zero-shot': {'p50_ms': 3134.5701454956725, 'p95_ms': 3784.149330600485}, 'fine-tuned': {'p50_ms': 1264.696083002491, 'p95_ms': 1577.265272902878}}
