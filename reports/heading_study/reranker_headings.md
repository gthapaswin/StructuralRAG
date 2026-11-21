# Heading study: the summary-section loss in a cross-encoder reranker

BAAI/bge-reranker-base (zero-shot) rescores the top-20 sections of flat retrieval within the document; MaxP windows of 200 words (stride 100). Role-neutral = summary-role sections read without their heading path (heading list, or label-free position rule). MRR@10; document-clustered tests.

| set | n | flat (first stage) | rerank: text only | rerank: path + text | role-neutral (list) | role-neutral (position) | heading gain: summary-only targets | heading gain: other targets |
|---|---|---|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 0.689 | 0.638 | 0.683 | 0.678 | 0.677 | +0.032 (n = 89) | +0.046 |
| GovReport (LLM-written) | 398 | 0.864 | 0.848 | 0.916 | 0.912 | 0.914 | +0.010 (n = 33) | +0.074 |
| QASPER (human) | 1256 | 0.509 | 0.460 | 0.522 | 0.570 | 0.575 | -0.045 (n = 198) | +0.082 |
| NQ (human) | 1744 | 0.642 | 0.680 | 0.704 | 0.704 | 0.705 | +0.019 (n = 940) | +0.029 |
| ConditionalQA (human) | 2422 | 0.700 | 0.732 | 0.783 | 0.776 | 0.766 | +0.059 (n = 74) | +0.051 |

| set | comparison | difference | 95% CI | p |
|---|---|---|---|---|
| GovReport (ICT) | path + text vs text only | +0.045 | [+0.030, +0.060] | 0.000 |
| GovReport (ICT) | role-neutral (list) vs path + text | -0.005 | [-0.010, -0.000] | 0.037 |
| GovReport (ICT) | role-neutral (position) vs path + text | -0.005 | [-0.009, -0.002] | 0.002 |
| GovReport (ICT) | path + text vs text only, summary targets | +0.032 | [-0.017, +0.081] | 0.237 |
| GovReport (LLM-written) | path + text vs text only | +0.068 | [+0.045, +0.093] | 0.000 |
| GovReport (LLM-written) | role-neutral (list) vs path + text | -0.004 | [-0.013, +0.004] | 0.408 |
| GovReport (LLM-written) | role-neutral (position) vs path + text | -0.002 | [-0.006, +0.001] | 0.375 |
| GovReport (LLM-written) | path + text vs text only, summary targets | +0.010 | [-0.051, +0.081] | 1.000 |
| QASPER (human) | path + text vs text only | +0.062 | [+0.043, +0.082] | 0.000 |
| QASPER (human) | role-neutral (list) vs path + text | +0.048 | [+0.036, +0.059] | 0.000 |
| QASPER (human) | role-neutral (position) vs path + text | +0.053 | [+0.045, +0.061] | 0.000 |
| QASPER (human) | path + text vs text only, summary targets | -0.045 | [-0.091, +0.001] | 0.080 |
| NQ (human) | path + text vs text only | +0.024 | [+0.014, +0.033] | 0.000 |
| NQ (human) | role-neutral (list) vs path + text | -0.000 | [-0.006, +0.005] | 0.853 |
| NQ (human) | role-neutral (position) vs path + text | +0.000 | [-0.005, +0.005] | 0.906 |
| NQ (human) | path + text vs text only, summary targets | +0.019 | [+0.006, +0.032] | 0.004 |
| ConditionalQA (human) | path + text vs text only | +0.051 | [+0.041, +0.061] | 0.000 |
| ConditionalQA (human) | role-neutral (list) vs path + text | -0.007 | [-0.011, -0.004] | 0.000 |
| ConditionalQA (human) | role-neutral (position) vs path + text | -0.017 | [-0.022, -0.011] | 0.000 |
| ConditionalQA (human) | path + text vs text only, summary targets | +0.059 | [+0.000, +0.118] | 0.068 |
