# Second domain: QASPER research papers

402 NLP papers, 6067 sections, 1256 human-written questions (gold = sections holding the annotator's evidence). Zero-shot transfer: settings and fine-tuned models come from GovReport; nothing is tuned on QASPER. Document scope.

| system | recall@1 | recall@5 | mrr@10 | ndcg@10 | p50_ms |
|---|---|---|---|---|---|
| bm25 | 0.225 | 0.751 | 0.439 | 0.552 | 1.245 |
| dense (base) | 0.349 | 0.752 | 0.516 | 0.605 | 4.979 |
| hybrid (base) | 0.316 | 0.781 | 0.506 | 0.606 | 7.968 |
| structure-aware (base) | 0.355 | 0.806 | 0.540 | 0.633 | 8.542 |
| dense (fine-tuned ICT) | 0.317 | 0.737 | 0.497 | 0.592 | 5.028 |
| hybrid (fine-tuned ICT) | 0.295 | 0.780 | 0.496 | 0.600 | 8.024 |
| structure-aware (fine-tuned ICT) | 0.317 | 0.737 | 0.497 | 0.592 | 7.616 |
| dense (fine-tuned ICT+LLM) | 0.344 | 0.757 | 0.521 | 0.611 | 4.980 |
| hybrid (fine-tuned ICT+LLM) | 0.312 | 0.793 | 0.511 | 0.612 | 7.748 |
| structure-aware (fine-tuned ICT+LLM) | 0.344 | 0.757 | 0.521 | 0.611 | 7.660 |
| + rerank MaxP (zero-shot) | 0.324 | 0.821 | 0.522 | 0.622 | 801.216 |
| + rerank MaxP (fine-tuned ICT) | 0.389 | 0.839 | 0.572 | 0.664 | 805.072 |
| + rerank MaxP (fine-tuned ICT+LLM) | 0.381 | 0.802 | 0.558 | 0.645 | 800.679 |


## Paired tests (MRR@10)

| system | vs | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| bm25 | dense (base) | -0.077 | 0.000 | 0.000 |
| dense (base) | bm25 | 0.077 | 0.000 | 0.000 |
| hybrid (base) | bm25 | 0.067 | 0.000 | 0.000 |
| hybrid (base) | dense (base) | -0.010 | 0.187 | 0.390 |
| hybrid (base) | structure-aware (base) | -0.035 | 0.000 | 0.000 |
| structure-aware (base) | bm25 | 0.102 | 0.000 | 0.000 |
| structure-aware (base) | dense (base) | 0.024 | 0.002 | 0.000 |
| dense (fine-tuned ICT) | bm25 | 0.058 | 0.000 | 0.000 |
| dense (fine-tuned ICT) | dense (base) | -0.019 | 0.008 | 0.031 |
| dense (fine-tuned ICT) | structure-aware (base) | -0.043 | 0.000 | 0.000 |
| hybrid (fine-tuned ICT) | bm25 | 0.057 | 0.000 | 0.000 |
| hybrid (fine-tuned ICT) | dense (base) | -0.020 | 0.026 | 0.219 |
| hybrid (fine-tuned ICT) | structure-aware (base) | -0.044 | 0.000 | 0.000 |
| structure-aware (fine-tuned ICT) | bm25 | 0.058 | 0.000 | 0.000 |
| structure-aware (fine-tuned ICT) | dense (base) | -0.019 | 0.008 | 0.031 |
| structure-aware (fine-tuned ICT) | structure-aware (base) | -0.043 | 0.000 | 0.000 |
| dense (fine-tuned ICT+LLM) | bm25 | 0.082 | 0.000 | 0.000 |
| dense (fine-tuned ICT+LLM) | dense (base) | 0.005 | 0.513 | 0.498 |
| dense (fine-tuned ICT+LLM) | structure-aware (base) | -0.020 | 0.021 | 0.001 |
| hybrid (fine-tuned ICT+LLM) | bm25 | 0.072 | 0.000 | 0.000 |
| hybrid (fine-tuned ICT+LLM) | dense (base) | -0.005 | 0.599 | 0.692 |
| hybrid (fine-tuned ICT+LLM) | structure-aware (base) | -0.029 | 0.000 | 0.001 |
| structure-aware (fine-tuned ICT+LLM) | bm25 | 0.082 | 0.000 | 0.000 |
| structure-aware (fine-tuned ICT+LLM) | dense (base) | 0.005 | 0.513 | 0.498 |
| structure-aware (fine-tuned ICT+LLM) | structure-aware (base) | -0.020 | 0.021 | 0.001 |
| + rerank MaxP (zero-shot) | bm25 | 0.084 | 0.000 | 0.000 |
| + rerank MaxP (zero-shot) | dense (base) | 0.006 | 0.586 | 0.443 |
| + rerank MaxP (zero-shot) | structure-aware (base) | -0.018 | 0.083 | 0.171 |
| + rerank MaxP (fine-tuned ICT) | bm25 | 0.134 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned ICT) | dense (base) | 0.056 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned ICT) | structure-aware (base) | 0.032 | 0.005 | 0.004 |
| + rerank MaxP (fine-tuned ICT+LLM) | bm25 | 0.119 | 0.000 | 0.000 |
| + rerank MaxP (fine-tuned ICT+LLM) | dense (base) | 0.042 | 0.001 | 0.000 |
| + rerank MaxP (fine-tuned ICT+LLM) | structure-aware (base) | 0.017 | 0.118 | 0.079 |


## MRR@10 by gold-section depth

| system | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| + rerank MaxP (fine-tuned ICT) | 0.583 | 0.553 | 0.506 | 0.267 |
| + rerank MaxP (fine-tuned ICT+LLM) | 0.549 | 0.576 | 0.626 | 0.238 |
| + rerank MaxP (zero-shot) | 0.523 | 0.522 | 0.522 | 0.238 |
| bm25 | 0.447 | 0.433 | 0.344 | 0.125 |
| dense (base) | 0.511 | 0.546 | 0.469 | 0.167 |
| dense (fine-tuned ICT) | 0.493 | 0.520 | 0.461 | 0.250 |
| dense (fine-tuned ICT+LLM) | 0.511 | 0.559 | 0.502 | 0.250 |
| hybrid (base) | 0.508 | 0.520 | 0.420 | 0.167 |
| hybrid (fine-tuned ICT) | 0.499 | 0.506 | 0.408 | 0.250 |
| hybrid (fine-tuned ICT+LLM) | 0.515 | 0.519 | 0.431 | 0.250 |
| structure-aware (base) | 0.534 | 0.577 | 0.480 | 0.167 |
| structure-aware (fine-tuned ICT) | 0.493 | 0.520 | 0.461 | 0.250 |
| structure-aware (fine-tuned ICT+LLM) | 0.511 | 0.559 | 0.502 | 0.250 |
