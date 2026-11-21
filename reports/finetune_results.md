# Fine-tuning results

Document scope (route a question to a section of a known report). Base encoder `BAAI/bge-small-en-v1.5`; fine-tuned on 1500 train-split reports with hard negatives from the same report, on ICT questions only or on ICT + LLM-written paraphrases. Rerankers rescore the top 20 of fine-tuned structure-aware; MaxP = best of overlapping 200-word windows. Structure weights re-tuned on dev per encoder: {'base': {'alpha': 0.2, 'beta': 0.0, 'sparse_weight': 1.0, 'rrf_k': 60}, 'finetuned': {'alpha': 0.0, 'beta': 0.0, 'sparse_weight': 0.0, 'rrf_k': 60}, 'finetuned_ICT+LLM': {'alpha': 0.0, 'beta': 0.0, 'sparse_weight': 0.0, 'rrf_k': 60}}.


## ICT questions (n = 1179)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | p50_ms |
|---|---|---|---|---|---|
| dense (base) | 0.511 | 0.872 | 0.661 | 0.731 | 5.006 |
| dense (fine-tuned) | 0.573 | 0.893 | 0.712 | 0.773 | 4.974 |
| dense (fine-tuned ICT+LLM) | 0.590 | 0.908 | 0.723 | 0.782 | 5.052 |
| hybrid (base) | 0.545 | 0.908 | 0.695 | 0.761 | 11.890 |
| hybrid (fine-tuned) | 0.596 | 0.913 | 0.731 | 0.791 | 12.173 |
| hybrid (fine-tuned ICT+LLM) | 0.601 | 0.915 | 0.736 | 0.794 | 11.759 |
| structure-aware (base) | 0.563 | 0.892 | 0.703 | 0.769 | 12.491 |
| structure-aware (fine-tuned) | 0.573 | 0.893 | 0.712 | 0.773 | 11.990 |
| structure-aware (fine-tuned ICT+LLM) | 0.590 | 0.908 | 0.723 | 0.782 | 12.326 |
| + rerank (zero-shot) | 0.499 | 0.862 | 0.648 | 0.721 | 530.549 |
| + rerank MaxP (zero-shot) | 0.537 | 0.885 | 0.680 | 0.749 | 1554.954 |
| + rerank (fine-tuned) | 0.558 | 0.878 | 0.690 | 0.754 | 537.614 |
| + rerank MaxP (fine-tuned) | 0.601 | 0.903 | 0.727 | 0.784 | 963.687 |
| + rerank (fine-tuned ICT+LLM) | 0.511 | 0.849 | 0.659 | 0.729 | 540.002 |
| + rerank MaxP (fine-tuned ICT+LLM) | 0.557 | 0.879 | 0.691 | 0.754 | 955.760 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| dense (fine-tuned) vs dense (base) | mrr@10 | 0.051 | 0.000 | 0.000 |
| dense (fine-tuned) vs dense (base) | recall@1 | 0.061 | 0.000 | 0.000 |
| dense (fine-tuned ICT+LLM) vs dense (base) | mrr@10 | 0.063 | 0.000 | 0.000 |
| dense (fine-tuned ICT+LLM) vs dense (base) | recall@1 | 0.079 | 0.000 | 0.000 |
| hybrid (fine-tuned) vs hybrid (base) | mrr@10 | 0.036 | 0.000 | 0.000 |
| hybrid (fine-tuned) vs hybrid (base) | recall@1 | 0.052 | 0.000 | 0.001 |
| hybrid (fine-tuned ICT+LLM) vs hybrid (base) | mrr@10 | 0.041 | 0.000 | 0.000 |
| hybrid (fine-tuned ICT+LLM) vs hybrid (base) | recall@1 | 0.056 | 0.000 | 0.001 |
| structure-aware (fine-tuned) vs structure-aware (base) | mrr@10 | 0.009 | 0.323 | 0.273 |
| structure-aware (fine-tuned) vs structure-aware (base) | recall@1 | 0.009 | 0.534 | 0.599 |
| structure-aware (fine-tuned ICT+LLM) vs structure-aware (base) | mrr@10 | 0.020 | 0.021 | 0.027 |
| structure-aware (fine-tuned ICT+LLM) vs structure-aware (base) | recall@1 | 0.027 | 0.053 | 0.124 |
| + rerank MaxP (zero-shot) vs + rerank (zero-shot) | mrr@10 | 0.032 | 0.000 | 0.144 |
| + rerank MaxP (zero-shot) vs + rerank (zero-shot) | recall@1 | 0.038 | 0.001 | 0.023 |
| + rerank MaxP (zero-shot) vs structure-aware (fine-tuned) | mrr@10 | -0.032 | 0.005 | 0.002 |
| + rerank MaxP (zero-shot) vs structure-aware (fine-tuned) | recall@1 | -0.036 | 0.032 | 0.062 |
| + rerank MaxP (fine-tuned) vs + rerank (fine-tuned) | mrr@10 | 0.037 | 0.000 | 0.003 |
| + rerank MaxP (fine-tuned) vs + rerank (fine-tuned) | recall@1 | 0.042 | 0.000 | 0.012 |
| + rerank MaxP (fine-tuned) vs structure-aware (fine-tuned) | mrr@10 | 0.015 | 0.172 | 0.320 |
| + rerank MaxP (fine-tuned) vs structure-aware (fine-tuned) | recall@1 | 0.028 | 0.097 | 0.139 |
| + rerank MaxP (fine-tuned) vs + rerank MaxP (zero-shot) | mrr@10 | 0.047 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned) vs + rerank MaxP (zero-shot) | recall@1 | 0.064 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank (fine-tuned ICT+LLM) | mrr@10 | 0.032 | 0.000 | 0.259 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank (fine-tuned ICT+LLM) | recall@1 | 0.046 | 0.000 | 0.006 |
| + rerank MaxP (fine-tuned ICT+LLM) vs structure-aware (fine-tuned) | mrr@10 | -0.021 | 0.060 | 0.030 |
| + rerank MaxP (fine-tuned ICT+LLM) vs structure-aware (fine-tuned) | recall@1 | -0.015 | 0.364 | 0.416 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank MaxP (zero-shot) | mrr@10 | 0.011 | 0.198 | 0.424 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank MaxP (zero-shot) | recall@1 | 0.020 | 0.143 | 0.248 |


## LLM-written questions (n = 398)

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | p50_ms |
|---|---|---|---|---|---|
| dense (base) | 0.791 | 0.980 | 0.871 | 0.903 | 5.758 |
| dense (fine-tuned) | 0.804 | 0.980 | 0.881 | 0.910 | 4.988 |
| dense (fine-tuned ICT+LLM) | 0.859 | 0.995 | 0.916 | 0.937 | 4.981 |
| hybrid (base) | 0.776 | 0.972 | 0.862 | 0.896 | 10.519 |
| hybrid (fine-tuned) | 0.779 | 0.980 | 0.866 | 0.899 | 10.854 |
| hybrid (fine-tuned ICT+LLM) | 0.804 | 0.985 | 0.883 | 0.912 | 10.433 |
| structure-aware (base) | 0.799 | 0.975 | 0.877 | 0.907 | 10.342 |
| structure-aware (fine-tuned) | 0.804 | 0.980 | 0.881 | 0.910 | 10.064 |
| structure-aware (fine-tuned ICT+LLM) | 0.859 | 0.995 | 0.916 | 0.937 | 10.466 |
| + rerank (zero-shot) | 0.854 | 0.992 | 0.916 | 0.937 | 555.036 |
| + rerank MaxP (zero-shot) | 0.862 | 0.987 | 0.917 | 0.937 | 1031.979 |
| + rerank (fine-tuned) | 0.791 | 0.977 | 0.874 | 0.905 | 554.234 |
| + rerank MaxP (fine-tuned) | 0.769 | 0.977 | 0.862 | 0.895 | 1034.724 |
| + rerank (fine-tuned ICT+LLM) | 0.907 | 0.990 | 0.943 | 0.957 | 552.786 |
| + rerank MaxP (fine-tuned ICT+LLM) | 0.894 | 0.992 | 0.937 | 0.953 | 1030.167 |



| comparison | metric | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| dense (fine-tuned) vs dense (base) | mrr@10 | 0.009 | 0.363 | 0.266 |
| dense (fine-tuned) vs dense (base) | recall@1 | 0.013 | 0.577 | 0.657 |
| dense (fine-tuned ICT+LLM) vs dense (base) | mrr@10 | 0.045 | 0.000 | 0.005 |
| dense (fine-tuned ICT+LLM) vs dense (base) | recall@1 | 0.068 | 0.000 | 0.018 |
| hybrid (fine-tuned) vs hybrid (base) | mrr@10 | 0.003 | 0.677 | 0.727 |
| hybrid (fine-tuned) vs hybrid (base) | recall@1 | 0.003 | 1.000 | 0.926 |
| hybrid (fine-tuned ICT+LLM) vs hybrid (base) | mrr@10 | 0.021 | 0.009 | 0.044 |
| hybrid (fine-tuned ICT+LLM) vs hybrid (base) | recall@1 | 0.028 | 0.072 | 0.306 |
| structure-aware (fine-tuned) vs structure-aware (base) | mrr@10 | 0.003 | 0.813 | 0.555 |
| structure-aware (fine-tuned) vs structure-aware (base) | recall@1 | 0.005 | 0.910 | 0.869 |
| structure-aware (fine-tuned ICT+LLM) vs structure-aware (base) | mrr@10 | 0.039 | 0.006 | 0.028 |
| structure-aware (fine-tuned ICT+LLM) vs structure-aware (base) | recall@1 | 0.060 | 0.014 | 0.049 |
| + rerank MaxP (zero-shot) vs + rerank (zero-shot) | mrr@10 | 0.001 | 0.909 | 0.673 |
| + rerank MaxP (zero-shot) vs + rerank (zero-shot) | recall@1 | 0.008 | 0.721 | 0.780 |
| + rerank MaxP (zero-shot) vs structure-aware (fine-tuned) | mrr@10 | 0.036 | 0.008 | 0.083 |
| + rerank MaxP (zero-shot) vs structure-aware (fine-tuned) | recall@1 | 0.058 | 0.011 | 0.052 |
| + rerank MaxP (fine-tuned) vs + rerank (fine-tuned) | mrr@10 | -0.012 | 0.216 | 0.123 |
| + rerank MaxP (fine-tuned) vs + rerank (fine-tuned) | recall@1 | -0.023 | 0.268 | 0.426 |
| + rerank MaxP (fine-tuned) vs structure-aware (fine-tuned) | mrr@10 | -0.019 | 0.173 | 0.216 |
| + rerank MaxP (fine-tuned) vs structure-aware (fine-tuned) | recall@1 | -0.035 | 0.177 | 0.253 |
| + rerank MaxP (fine-tuned) vs + rerank MaxP (zero-shot) | mrr@10 | -0.055 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned) vs + rerank MaxP (zero-shot) | recall@1 | -0.093 | 0.000 | 0.002 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank (fine-tuned ICT+LLM) | mrr@10 | -0.006 | 0.425 | 0.321 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank (fine-tuned ICT+LLM) | recall@1 | -0.013 | 0.427 | 0.637 |
| + rerank MaxP (fine-tuned ICT+LLM) vs structure-aware (fine-tuned) | mrr@10 | 0.057 | 0.000 | 0.003 |
| + rerank MaxP (fine-tuned ICT+LLM) vs structure-aware (fine-tuned) | recall@1 | 0.090 | 0.000 | 0.003 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank MaxP (zero-shot) | mrr@10 | 0.021 | 0.048 | 0.161 |
| + rerank MaxP (fine-tuned ICT+LLM) vs + rerank MaxP (zero-shot) | recall@1 | 0.033 | 0.093 | 0.248 |
