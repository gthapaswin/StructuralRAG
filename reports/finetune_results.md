# Fine-tuning results

Document scope (route a question to a section of a known report). Base encoder `BAAI/bge-small-en-v1.5`; fine-tuned on 1500 train-split reports (ICT questions, hard negatives from the same report). Structure-aware weights re-tuned on dev for each encoder: base {'alpha': 0.2, 'beta': 0.0, 'sparse_weight': 1.0, 'rrf_k': 60}, fine-tuned {'alpha': 0.0, 'beta': 0.0, 'sparse_weight': 0.0, 'rrf_k': 60}.


## ICT questions (n = 1179)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | p50_ms |
|---|---|---|---|---|---|
| dense (base) | 0.511 | 0.872 | 0.661 | 0.731 | 21.081 |
| dense (fine-tuned) | 0.573 | 0.893 | 0.712 | 0.773 | 18.354 |
| hybrid (base) | 0.545 | 0.908 | 0.695 | 0.761 | 25.810 |
| hybrid (fine-tuned) | 0.596 | 0.913 | 0.731 | 0.791 | 26.539 |
| structure-aware (base) | 0.563 | 0.892 | 0.703 | 0.769 | 26.984 |
| structure-aware (fine-tuned) | 0.573 | 0.893 | 0.712 | 0.773 | 27.679 |
| + rerank (zero-shot) | 0.499 | 0.862 | 0.648 | 0.721 | 750.373 |
| + rerank (fine-tuned) | 0.558 | 0.878 | 0.690 | 0.754 | 740.234 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| dense (fine-tuned) vs dense (base) | mrr@10 | 0.051 | 0.000 | 0.000 |
| dense (fine-tuned) vs dense (base) | recall@1 | 0.061 | 0.000 | 0.000 |
| hybrid (fine-tuned) vs hybrid (base) | mrr@10 | 0.036 | 0.000 | 0.000 |
| hybrid (fine-tuned) vs hybrid (base) | recall@1 | 0.052 | 0.000 | 0.001 |
| structure-aware (fine-tuned) vs structure-aware (base) | mrr@10 | 0.009 | 0.323 | 0.273 |
| structure-aware (fine-tuned) vs structure-aware (base) | recall@1 | 0.009 | 0.534 | 0.599 |
| + rerank (zero-shot) vs structure-aware (fine-tuned) | mrr@10 | -0.064 | 0.000 | 0.000 |
| + rerank (zero-shot) vs structure-aware (fine-tuned) | recall@1 | -0.074 | 0.000 | 0.000 |
| + rerank (fine-tuned) vs structure-aware (fine-tuned) | mrr@10 | -0.022 | 0.034 | 0.025 |
| + rerank (fine-tuned) vs structure-aware (fine-tuned) | recall@1 | -0.014 | 0.386 | 0.438 |
| + rerank (fine-tuned) vs + rerank (zero-shot) | mrr@10 | 0.042 | 0.000 | 0.000 |
| + rerank (fine-tuned) vs + rerank (zero-shot) | recall@1 | 0.059 | 0.000 | 0.001 |


## LLM-written questions (n = 398)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | p50_ms |
|---|---|---|---|---|---|
| dense (base) | 0.791 | 0.980 | 0.871 | 0.903 | 18.380 |
| dense (fine-tuned) | 0.804 | 0.980 | 0.881 | 0.910 | 17.863 |
| hybrid (base) | 0.776 | 0.972 | 0.862 | 0.896 | 25.585 |
| hybrid (fine-tuned) | 0.779 | 0.980 | 0.866 | 0.899 | 25.687 |
| structure-aware (base) | 0.799 | 0.975 | 0.877 | 0.907 | 24.407 |
| structure-aware (fine-tuned) | 0.804 | 0.980 | 0.881 | 0.910 | 24.688 |
| + rerank (zero-shot) | 0.854 | 0.992 | 0.916 | 0.937 | 768.531 |
| + rerank (fine-tuned) | 0.791 | 0.977 | 0.874 | 0.905 | 774.329 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| dense (fine-tuned) vs dense (base) | mrr@10 | 0.009 | 0.363 | 0.266 |
| dense (fine-tuned) vs dense (base) | recall@1 | 0.013 | 0.577 | 0.657 |
| hybrid (fine-tuned) vs hybrid (base) | mrr@10 | 0.003 | 0.677 | 0.727 |
| hybrid (fine-tuned) vs hybrid (base) | recall@1 | 0.003 | 1.000 | 0.926 |
| structure-aware (fine-tuned) vs structure-aware (base) | mrr@10 | 0.003 | 0.813 | 0.555 |
| structure-aware (fine-tuned) vs structure-aware (base) | recall@1 | 0.005 | 0.910 | 0.869 |
| + rerank (zero-shot) vs structure-aware (fine-tuned) | mrr@10 | 0.035 | 0.005 | 0.044 |
| + rerank (zero-shot) vs structure-aware (fine-tuned) | recall@1 | 0.050 | 0.024 | 0.089 |
| + rerank (fine-tuned) vs structure-aware (fine-tuned) | mrr@10 | -0.007 | 0.614 | 0.541 |
| + rerank (fine-tuned) vs structure-aware (fine-tuned) | recall@1 | -0.013 | 0.657 | 0.676 |
| + rerank (fine-tuned) vs + rerank (zero-shot) | mrr@10 | -0.042 | 0.000 | 0.007 |
| + rerank (fine-tuned) vs + rerank (zero-shot) | recall@1 | -0.063 | 0.003 | 0.032 |
