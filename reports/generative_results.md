# Generative retrieval and LLM-written questions

LLM: `qwen3:8b` via Ollama (temperature 0, output constrained to the report's section ids). Document scope. Structure-aware weights from the main evaluation (α=0.2, β=0.0).


## ICT queries (sentence from the section) — 300 queries

| method | recall@1 | recall@5 | mrr@10 | ndcg@10 | same_branch | mean_returned |
|---|---|---|---|---|---|---|
| bm25 | 0.547 | 0.887 | 0.685 | 0.751 | 0.837 | 9.700 |
| dense | 0.520 | 0.863 | 0.668 | 0.738 | 0.787 | 9.700 |
| hybrid | 0.553 | 0.913 | 0.701 | 0.766 | 0.813 | 9.700 |
| structure_aware | 0.563 | 0.893 | 0.706 | 0.771 | 0.823 | 9.700 |
| generative_toc | 0.337 | 0.623 | 0.445 | 0.489 | 0.770 | 4.370 |
| generative_toc_snippets | 0.380 | 0.723 | 0.510 | 0.563 | 0.753 | 4.420 |
| generative+structure | 0.503 | 0.777 | 0.641 | 0.718 | 0.797 | 9.700 |


**Paired tests**

| method | vs | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|---|
| generative_toc_snippets | structure_aware | mrr@10 | -0.196 | 0.000 | 0.000 |
| generative_toc_snippets | structure_aware | recall@1 | -0.183 | 0.000 | 0.000 |
| generative+structure | structure_aware | mrr@10 | -0.065 | 0.001 | 0.002 |
| generative+structure | structure_aware | recall@1 | -0.060 | 0.048 | 0.095 |
| generative+structure | hybrid | mrr@10 | -0.060 | 0.004 | 0.003 |
| generative+structure | hybrid | recall@1 | -0.050 | 0.119 | 0.169 |
| structure_aware | hybrid | mrr@10 | 0.005 | 0.622 | 0.780 |
| structure_aware | hybrid | recall@1 | 0.010 | 0.708 | 0.752 |
| structure_aware | bm25 | mrr@10 | 0.021 | 0.156 | 0.134 |
| structure_aware | bm25 | recall@1 | 0.017 | 0.589 | 0.625 |
| structure_aware | dense | mrr@10 | 0.037 | 0.028 | 0.039 |
| structure_aware | dense | recall@1 | 0.043 | 0.099 | 0.201 |


**MRR@10 by target depth**

| method | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|
| bm25 | 0.701 | 0.719 | 0.603 | 0.663 | 0.556 |
| dense | 0.654 | 0.699 | 0.599 | 0.699 | 0.536 |
| generative+structure | 0.542 | 0.687 | 0.578 | 0.734 | 0.781 |
| generative_toc | 0.408 | 0.472 | 0.345 | 0.578 | 0.333 |
| generative_toc_snippets | 0.375 | 0.570 | 0.434 | 0.657 | 0.458 |
| hybrid | 0.712 | 0.723 | 0.643 | 0.694 | 0.594 |
| structure_aware | 0.704 | 0.726 | 0.664 | 0.708 | 0.594 |


## LLM-written paraphrased questions — 398 queries

| method | recall@1 | recall@5 | mrr@10 | ndcg@10 | same_branch | mean_returned |
|---|---|---|---|---|---|---|
| bm25 | 0.724 | 0.945 | 0.815 | 0.857 | 0.864 | 9.606 |
| dense | 0.791 | 0.980 | 0.871 | 0.903 | 0.902 | 9.606 |
| hybrid | 0.776 | 0.972 | 0.862 | 0.896 | 0.884 | 9.606 |
| structure_aware | 0.799 | 0.975 | 0.877 | 0.907 | 0.910 | 9.606 |
| generative_toc | 0.545 | 0.776 | 0.644 | 0.678 | 0.847 | 4.291 |
| generative_toc_snippets | 0.641 | 0.864 | 0.732 | 0.766 | 0.857 | 4.304 |
| generative+structure | 0.751 | 0.872 | 0.826 | 0.867 | 0.892 | 9.606 |


**Paired tests**

| method | vs | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|---|
| generative_toc_snippets | structure_aware | mrr@10 | -0.145 | 0.000 | 0.000 |
| generative_toc_snippets | structure_aware | recall@1 | -0.158 | 0.000 | 0.000 |
| generative+structure | structure_aware | mrr@10 | -0.051 | 0.001 | 0.003 |
| generative+structure | structure_aware | recall@1 | -0.048 | 0.048 | 0.114 |
| generative+structure | hybrid | mrr@10 | -0.036 | 0.041 | 0.091 |
| generative+structure | hybrid | recall@1 | -0.025 | 0.379 | 0.423 |
| structure_aware | hybrid | mrr@10 | 0.015 | 0.130 | 0.263 |
| structure_aware | hybrid | recall@1 | 0.023 | 0.239 | 0.420 |
| structure_aware | bm25 | mrr@10 | 0.062 | 0.000 | 0.000 |
| structure_aware | bm25 | recall@1 | 0.075 | 0.000 | 0.009 |
| structure_aware | dense | mrr@10 | 0.006 | 0.653 | 0.970 |
| structure_aware | dense | recall@1 | 0.008 | 0.828 | 0.802 |


**MRR@10 by target depth**

| method | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|
| bm25 | 0.802 | 0.825 | 0.795 | 0.852 | 0.770 |
| dense | 0.852 | 0.884 | 0.872 | 0.874 | 0.810 |
| generative+structure | 0.710 | 0.863 | 0.893 | 0.821 | 0.929 |
| generative_toc | 0.576 | 0.647 | 0.703 | 0.672 | 0.786 |
| generative_toc_snippets | 0.607 | 0.757 | 0.811 | 0.772 | 0.886 |
| hybrid | 0.854 | 0.866 | 0.864 | 0.869 | 0.821 |
| structure_aware | 0.819 | 0.892 | 0.906 | 0.915 | 0.821 |


## Question generation

```
{'generated': 602, 'accepted': 398, 'rejected_copies_text': 139, 'rejected_unanswerable': 54, 'rejected_mentions_section': 11, 'seconds': 1920.3, 'model': 'qwen3:8b'}
```


Generative latency (uncached, ToC+snippets): {'mean_ms': 2128.897187500297, 'p50_ms': 1871.8093960014812, 'p95_ms': 3808.688563949545}
