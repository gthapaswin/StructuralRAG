# Heading study: label-free summary roles

Which sections get a neutral heading score (path score = the document mean). 'Heading list' is the role-neutral method (fixed English list); the other rules never read heading text. Rule chosen on GovReport dev: **position+last**. MRR@10, alpha = 0.2, bge-small.

| set | n | flat | SA | heading list | position | position+last | centrality |
|---|---|---|---|---|---|---|---|
| GovReport dev (ICT, selection) | 355 | 0.698 | 0.703 | 0.703 | 0.703 | 0.704 | 0.701 |
| GovReport (ICT) | 1179 | 0.689 | 0.691 | 0.692 | 0.692 | 0.693 | 0.692 |
| GovReport (LLM-written) | 398 | 0.864 | 0.878 | 0.886 | 0.881 | 0.881 | 0.872 |
| QASPER (human) | 1256 | 0.509 | 0.542 | 0.537 | 0.536 | 0.534 | 0.539 |
| NQ (human) | 1744 | 0.642 | 0.616 | 0.627 | 0.627 | 0.627 | 0.619 |
| ConditionalQA (human) | 2422 | 0.700 | 0.695 | 0.706 | 0.706 | 0.705 | 0.700 |

How the label-free rules relate to the heading list (test documents):

| set | rule | sections neutralized | of which on the list | list sections found |
|---|---|---|---|---|
| GovReport dev (ICT, selection) | position | 60 | 58% | 36% |
| GovReport dev (ICT, selection) | position+last | 97 | 36% | 36% |
| GovReport dev (ICT, selection) | centrality | 60 | 23% | 15% |
| GovReport (ICT) | position | 200 | 56% | 39% |
| GovReport (ICT) | position+last | 322 | 35% | 39% |
| GovReport (ICT) | centrality | 200 | 20% | 14% |
| QASPER (human) | position | 402 | 100% | 33% |
| QASPER (human) | position+last | 804 | 71% | 47% |
| QASPER (human) | centrality | 402 | 52% | 17% |
| NQ (human) | position | 1744 | 100% | 88% |
| NQ (human) | position+last | 3488 | 50% | 88% |
| NQ (human) | centrality | 1744 | 7% | 6% |
| ConditionalQA (human) | position | 495 | 63% | 99% |
| ConditionalQA (human) | position+last | 990 | 32% | 99% |
| ConditionalQA (human) | centrality | 495 | 14% | 23% |

Document-clustered tests for the chosen rule (position+last) and the heading list:

| set | comparison | difference | 95% CI | p |
|---|---|---|---|---|
| GovReport dev (ICT, selection) | position+last vs SA | +0.002 | [-0.001, +0.005] | 0.436 |
| GovReport dev (ICT, selection) | position+last vs flat | +0.006 | [-0.014, +0.027] | 0.623 |
| GovReport dev (ICT, selection) | position+last vs heading list | +0.001 | [-0.002, +0.005] | 0.603 |
| GovReport dev (ICT, selection) | heading list vs SA | +0.000 | [-0.001, +0.002] | 0.623 |
| GovReport (ICT) | position+last vs SA | +0.002 | [-0.001, +0.004] | 0.150 |
| GovReport (ICT) | position+last vs flat | +0.003 | [-0.007, +0.014] | 0.524 |
| GovReport (ICT) | position+last vs heading list | +0.001 | [-0.004, +0.005] | 0.800 |
| GovReport (ICT) | heading list vs SA | +0.001 | [-0.003, +0.005] | 0.518 |
| GovReport (LLM-written) | position+last vs SA | +0.003 | [-0.000, +0.007] | 0.254 |
| GovReport (LLM-written) | position+last vs flat | +0.017 | [-0.000, +0.035] | 0.061 |
| GovReport (LLM-written) | position+last vs heading list | -0.005 | [-0.013, +0.002] | 0.175 |
| GovReport (LLM-written) | heading list vs SA | +0.008 | [+0.001, +0.016] | 0.024 |
| QASPER (human) | position+last vs SA | -0.008 | [-0.012, -0.004] | 0.000 |
| QASPER (human) | position+last vs flat | +0.025 | [+0.013, +0.036] | 0.000 |
| QASPER (human) | position+last vs heading list | -0.003 | [-0.007, +0.001] | 0.105 |
| QASPER (human) | heading list vs SA | -0.005 | [-0.010, -0.000] | 0.077 |
| NQ (human) | position+last vs SA | +0.011 | [+0.006, +0.015] | 0.000 |
| NQ (human) | position+last vs flat | -0.015 | [-0.025, -0.006] | 0.002 |
| NQ (human) | position+last vs heading list | -0.000 | [-0.001, +0.001] | 0.983 |
| NQ (human) | heading list vs SA | +0.011 | [+0.006, +0.015] | 0.000 |
| ConditionalQA (human) | position+last vs SA | +0.009 | [+0.005, +0.014] | 0.000 |
| ConditionalQA (human) | position+last vs flat | +0.004 | [-0.006, +0.015] | 0.425 |
| ConditionalQA (human) | position+last vs heading list | -0.001 | [-0.004, +0.002] | 0.444 |
| ConditionalQA (human) | heading list vs SA | +0.011 | [+0.007, +0.014] | 0.000 |
