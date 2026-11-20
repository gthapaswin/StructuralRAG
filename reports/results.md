# StructuralRAG benchmark results

Dataset: GovReport (structure), test docs=200, sections=4533, queries=1179 (ICT). Embedding model: `BAAI/bge-small-en-v1.5`.

Tuned on dev: alpha=0.2, beta=0.0, sparse_weight=1.0


## Scope: document

| method | n_queries | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | same_branch | latency_p50_ms | latency_p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25[text] | 1179 | 0.546 | 0.880 | 0.956 | 0.687 | 0.753 | 0.817 | 4.434 | 7.380 |
| dense[text] | 1179 | 0.511 | 0.872 | 0.950 | 0.661 | 0.731 | 0.785 | 5.285 | 6.858 |
| hybrid[text] | 1179 | 0.545 | 0.908 | 0.967 | 0.695 | 0.761 | 0.795 | 12.442 | 15.992 |
| structure_aware | 1179 | 0.563 | 0.892 | 0.975 | 0.703 | 0.769 | 0.814 | 12.670 | 16.274 |


## Scope: corpus

| method | n_queries | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | same_branch | latency_p50_ms | latency_p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25[text] | 1179 | 0.451 | 0.749 | 0.819 | 0.575 | 0.634 | 0.675 | 4.445 | 7.576 |
| dense[text] | 1179 | 0.399 | 0.683 | 0.773 | 0.520 | 0.581 | 0.611 | 5.230 | 6.069 |
| hybrid[text] | 1179 | 0.425 | 0.765 | 0.832 | 0.565 | 0.630 | 0.643 | 12.168 | 15.641 |
| structure_aware | 1179 | 0.444 | 0.751 | 0.839 | 0.573 | 0.637 | 0.664 | 13.239 | 17.197 |


## Structure-aware vs baselines (paired, per query)

| scope | baseline | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|---|
| document | bm25 | mrr@10 | 0.015 | 0.064 | 0.003 |
| document | bm25 | recall@1 | 0.017 | 0.208 | 0.327 |
| document | dense | mrr@10 | 0.042 | 0.000 | 0.000 |
| document | dense | recall@1 | 0.052 | 0.000 | 0.003 |
| document | hybrid | mrr@10 | 0.008 | 0.139 | 0.634 |
| document | hybrid | recall@1 | 0.019 | 0.061 | 0.246 |
| corpus | bm25 | mrr@10 | -0.002 | 0.841 | 0.022 |
| corpus | bm25 | recall@1 | -0.007 | 0.632 | 0.691 |
| corpus | dense | mrr@10 | 0.054 | 0.000 | 0.000 |
| corpus | dense | recall@1 | 0.046 | 0.000 | 0.007 |
| corpus | hybrid | mrr@10 | 0.008 | 0.135 | 0.684 |
| corpus | hybrid | recall@1 | 0.020 | 0.041 | 0.223 |


## Ablation: text views (document scope)

| view | method | recall@1 | recall@5 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|
| text | bm25 | 0.546 | 0.880 | 0.687 | 0.753 |
| text | dense | 0.511 | 0.872 | 0.661 | 0.731 |
| text | hybrid | 0.545 | 0.908 | 0.695 | 0.761 |
| title | bm25 | 0.556 | 0.884 | 0.694 | 0.758 |
| title | dense | 0.494 | 0.865 | 0.646 | 0.720 |
| title | hybrid | 0.541 | 0.903 | 0.693 | 0.760 |
| path | bm25 | 0.550 | 0.880 | 0.688 | 0.754 |
| path | dense | 0.502 | 0.862 | 0.650 | 0.722 |
| path | hybrid | 0.550 | 0.902 | 0.696 | 0.762 |
| parent | bm25 | 0.549 | 0.873 | 0.687 | 0.754 |
| parent | dense | 0.492 | 0.852 | 0.639 | 0.713 |
| parent | hybrid | 0.542 | 0.889 | 0.690 | 0.759 |
| full | bm25 | 0.551 | 0.871 | 0.686 | 0.752 |
| full | dense | 0.489 | 0.836 | 0.632 | 0.705 |
| full | hybrid | 0.539 | 0.887 | 0.685 | 0.754 |


## Ablation: structure-aware score components (document scope)

| configuration | recall@1 | recall@5 | mrr@10 | ndcg@10 |
|---|---|---|---|---|
| content only | 0.511 | 0.872 | 0.661 | 0.731 |
| + path | 0.520 | 0.859 | 0.663 | 0.733 |
| + parent | 0.511 | 0.872 | 0.661 | 0.731 |
| + path + parent | 0.520 | 0.859 | 0.663 | 0.733 |
| + path + parent + BM25(full) | 0.563 | 0.892 | 0.703 | 0.769 |


## MRR@10 by target section depth (document scope)

| method | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| bm25 | 0.702 | 0.715 | 0.633 | 0.673 | 0.560 |
| dense | 0.703 | 0.654 | 0.629 | 0.650 | 0.599 |
| hybrid | 0.708 | 0.704 | 0.672 | 0.676 | 0.653 |
| structure_aware | 0.686 | 0.715 | 0.700 | 0.704 | 0.699 |



## Efficiency

```
{'index_build_seconds': {'bm25': 0.25131633300043177, 'dense': 25.973388375001377, 'hybrid': 1.2250002328073606e-05, 'structure_aware': 3.4538557920022868, 'total_s': 29.67858120800156}, 'embedding_storage_mb': 41.776128, 'peak_rss_mb': 885.751808, 'corpus_sections': 4533}
```
