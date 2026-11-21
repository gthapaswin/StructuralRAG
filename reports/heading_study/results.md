# Heading study

Within-document section retrieval. Structure-aware (SA) uses the GovReport-dev settings (path weight alpha = 0.2, BM25 over the structural view fused by RRF); flat = RRF of BM25 and dense over section text only. Only headings change between variants; section text and questions are identical.


## A. Do headings help? (MRR@10)

| set | n | flat | SA real | SA placebo | SA generic | SA flattened | SA OCR 0.2 |
|---|---|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 0.689 | 0.691 | 0.690 | 0.689 | 0.693 | 0.694 |
| GovReport (LLM-written) | 398 | 0.864 | 0.878 | 0.866 | 0.867 | 0.883 | 0.855 |
| QASPER (human) | 1256 | 0.509 | 0.542 | 0.507 | 0.507 | 0.539 | 0.524 |
| NQ (human) | 1744 | 0.640 | 0.616 | 0.624 | 0.629 | 0.617 | 0.624 |



| set | comparison | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| GovReport (ICT) | SA real vs flat | 0.002 | 0.767 | 0.934 |
| GovReport (ICT) | SA real vs SA placebo | 0.000 | 0.911 | 0.855 |
| GovReport (ICT) | SA real vs SA generic | 0.002 | 0.662 | 0.627 |
| GovReport (LLM-written) | SA real vs flat | 0.014 | 0.123 | 0.292 |
| GovReport (LLM-written) | SA real vs SA placebo | 0.012 | 0.115 | 0.099 |
| GovReport (LLM-written) | SA real vs SA generic | 0.011 | 0.170 | 0.329 |
| QASPER (human) | SA real vs flat | 0.033 | 0.000 | 0.000 |
| QASPER (human) | SA real vs SA placebo | 0.035 | 0.000 | 0.000 |
| QASPER (human) | SA real vs SA generic | 0.034 | 0.000 | 0.000 |
| NQ (human) | SA real vs flat | -0.023 | 0.000 | 0.000 |
| NQ (human) | SA real vs SA placebo | -0.007 | 0.041 | 0.413 |
| NQ (human) | SA real vs SA generic | -0.013 | 0.001 | 0.184 |


## B. Dose-response (SA gain over flat, MRR@10)

| set | corruption | level | gain_vs_flat | ci_lo | ci_hi |
|---|---|---|---|---|---|
| GovReport (ICT) | generic | 0.000 | 0.002 | -0.009 | 0.013 |
| GovReport (ICT) | generic | 0.250 | 0.001 | -0.011 | 0.011 |
| GovReport (ICT) | generic | 0.500 | -0.004 | -0.015 | 0.006 |
| GovReport (ICT) | generic | 0.750 | -0.004 | -0.013 | 0.005 |
| GovReport (ICT) | generic | 1.000 | -0.001 | -0.007 | 0.005 |
| GovReport (ICT) | ocr | 0.000 | 0.002 | -0.009 | 0.013 |
| GovReport (ICT) | ocr | 0.100 | -0.001 | -0.012 | 0.010 |
| GovReport (ICT) | ocr | 0.200 | 0.004 | -0.006 | 0.014 |
| GovReport (ICT) | ocr | 0.400 | -0.000 | -0.009 | 0.009 |
| GovReport (LLM-written) | generic | 0.000 | 0.014 | -0.003 | 0.033 |
| GovReport (LLM-written) | generic | 0.250 | 0.005 | -0.013 | 0.022 |
| GovReport (LLM-written) | generic | 0.500 | 0.002 | -0.016 | 0.019 |
| GovReport (LLM-written) | generic | 0.750 | -0.009 | -0.026 | 0.008 |
| GovReport (LLM-written) | generic | 1.000 | 0.003 | -0.007 | 0.014 |
| GovReport (LLM-written) | ocr | 0.000 | 0.014 | -0.003 | 0.033 |
| GovReport (LLM-written) | ocr | 0.100 | 0.001 | -0.017 | 0.019 |
| GovReport (LLM-written) | ocr | 0.200 | -0.008 | -0.024 | 0.007 |
| GovReport (LLM-written) | ocr | 0.400 | -0.001 | -0.015 | 0.011 |
| QASPER (human) | generic | 0.000 | 0.033 | 0.021 | 0.044 |
| QASPER (human) | generic | 0.250 | 0.021 | 0.010 | 0.032 |
| QASPER (human) | generic | 0.500 | 0.010 | 0.000 | 0.020 |
| QASPER (human) | generic | 0.750 | 0.008 | -0.001 | 0.018 |
| QASPER (human) | generic | 1.000 | -0.002 | -0.010 | 0.006 |
| QASPER (human) | ocr | 0.000 | 0.033 | 0.021 | 0.044 |
| QASPER (human) | ocr | 0.100 | 0.021 | 0.009 | 0.031 |
| QASPER (human) | ocr | 0.200 | 0.015 | 0.004 | 0.025 |
| QASPER (human) | ocr | 0.400 | 0.004 | -0.005 | 0.013 |
| NQ (human) | generic | 0.000 | -0.023 | -0.033 | -0.014 |
| NQ (human) | generic | 0.250 | -0.017 | -0.026 | -0.007 |
| NQ (human) | generic | 0.500 | -0.016 | -0.025 | -0.007 |
| NQ (human) | generic | 0.750 | -0.014 | -0.022 | -0.006 |
| NQ (human) | generic | 1.000 | -0.010 | -0.018 | -0.003 |
| NQ (human) | ocr | 0.000 | -0.023 | -0.033 | -0.014 |
| NQ (human) | ocr | 0.100 | -0.020 | -0.029 | -0.012 |
| NQ (human) | ocr | 0.200 | -0.016 | -0.024 | -0.007 |
| NQ (human) | ocr | 0.400 | -0.018 | -0.026 | -0.010 |


## C. Does heading informativeness predict the gain?

Per document (real headings): Spearman correlation between each feature and the document's mean gain.

| set | feature | target | docs | spearman | ci_lo | ci_hi | p |
|---|---|---|---|---|---|---|---|
| GovReport (ICT) | self_nrank | gain | 200 | 0.130 | -0.003 | 0.253 | 0.066 |
| GovReport (ICT) | self_nrank | gain_vs_placebo | 200 | 0.076 | -0.052 | 0.208 | 0.286 |
| GovReport (ICT) | title_mrr | gain | 200 | 0.127 | -0.014 | 0.264 | 0.074 |
| GovReport (ICT) | title_mrr | gain_vs_placebo | 200 | 0.070 | -0.064 | 0.199 | 0.322 |
| GovReport (ICT) | sibling_sim | gain | 200 | 0.064 | -0.075 | 0.194 | 0.366 |
| GovReport (ICT) | sibling_sim | gain_vs_placebo | 200 | 0.029 | -0.107 | 0.164 | 0.687 |
| GovReport (ICT) | generic_rate | gain | 200 | -0.023 | -0.162 | 0.119 | 0.746 |
| GovReport (ICT) | generic_rate | gain_vs_placebo | 200 | -0.085 | -0.219 | 0.046 | 0.232 |
| GovReport (ICT) | title_idf | gain | 200 | 0.023 | -0.113 | 0.157 | 0.748 |
| GovReport (ICT) | title_idf | gain_vs_placebo | 200 | 0.088 | -0.039 | 0.213 | 0.216 |
| GovReport (LLM-written) | self_nrank | gain | 200 | -0.066 | -0.203 | 0.075 | 0.356 |
| GovReport (LLM-written) | self_nrank | gain_vs_placebo | 200 | -0.100 | -0.228 | 0.028 | 0.161 |
| GovReport (LLM-written) | title_mrr | gain | 200 | -0.123 | -0.249 | 0.020 | 0.084 |
| GovReport (LLM-written) | title_mrr | gain_vs_placebo | 200 | -0.105 | -0.227 | 0.039 | 0.140 |
| GovReport (LLM-written) | sibling_sim | gain | 200 | 0.058 | -0.082 | 0.201 | 0.415 |
| GovReport (LLM-written) | sibling_sim | gain_vs_placebo | 200 | 0.033 | -0.101 | 0.179 | 0.642 |
| GovReport (LLM-written) | generic_rate | gain | 200 | -0.023 | -0.183 | 0.114 | 0.745 |
| GovReport (LLM-written) | generic_rate | gain_vs_placebo | 200 | 0.047 | -0.098 | 0.181 | 0.512 |
| GovReport (LLM-written) | title_idf | gain | 200 | -0.077 | -0.224 | 0.089 | 0.280 |
| GovReport (LLM-written) | title_idf | gain_vs_placebo | 200 | -0.167 | -0.301 | -0.020 | 0.018 |
| QASPER (human) | self_nrank | gain | 402 | -0.011 | -0.110 | 0.078 | 0.821 |
| QASPER (human) | self_nrank | gain_vs_placebo | 402 | 0.008 | -0.098 | 0.110 | 0.868 |
| QASPER (human) | title_mrr | gain | 402 | -0.021 | -0.115 | 0.073 | 0.671 |
| QASPER (human) | title_mrr | gain_vs_placebo | 402 | 0.027 | -0.067 | 0.121 | 0.596 |
| QASPER (human) | sibling_sim | gain | 402 | -0.023 | -0.114 | 0.078 | 0.647 |
| QASPER (human) | sibling_sim | gain_vs_placebo | 402 | 0.028 | -0.064 | 0.121 | 0.582 |
| QASPER (human) | generic_rate | gain | 402 | 0.006 | -0.097 | 0.108 | 0.907 |
| QASPER (human) | generic_rate | gain_vs_placebo | 402 | 0.036 | -0.055 | 0.134 | 0.475 |
| QASPER (human) | title_idf | gain | 402 | -0.010 | -0.115 | 0.090 | 0.838 |
| QASPER (human) | title_idf | gain_vs_placebo | 402 | -0.046 | -0.148 | 0.046 | 0.360 |
| NQ (human) | self_nrank | gain | 1744 | -0.002 | -0.050 | 0.042 | 0.939 |
| NQ (human) | self_nrank | gain_vs_placebo | 1744 | -0.068 | -0.111 | -0.019 | 0.005 |
| NQ (human) | title_mrr | gain | 1744 | -0.009 | -0.058 | 0.036 | 0.718 |
| NQ (human) | title_mrr | gain_vs_placebo | 1744 | -0.039 | -0.084 | 0.011 | 0.102 |
| NQ (human) | sibling_sim | gain | 1744 | 0.036 | -0.010 | 0.082 | 0.129 |
| NQ (human) | sibling_sim | gain_vs_placebo | 1744 | 0.056 | 0.010 | 0.107 | 0.019 |
| NQ (human) | generic_rate | gain | 1744 | 0.017 | -0.029 | 0.064 | 0.471 |
| NQ (human) | generic_rate | gain_vs_placebo | 1744 | 0.116 | 0.068 | 0.159 | 0.000 |
| NQ (human) | title_idf | gain | 1744 | -0.017 | -0.062 | 0.030 | 0.481 |
| NQ (human) | title_idf | gain_vs_placebo | 1744 | -0.097 | -0.141 | -0.049 | 0.000 |


Per question: does the gold section's own heading path rank it first among its document's sections?

| set | questions | spearman | ci_lo | ci_hi | p |
|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | -0.006 | -0.062 | 0.047 | 0.833 |
| GovReport (LLM-written) | 398 | 0.092 | -0.008 | 0.189 | 0.066 |
| QASPER (human) | 1256 | -0.002 | -0.058 | 0.051 | 0.933 |
| NQ (human) | 1744 | -0.028 | -0.072 | 0.014 | 0.243 |


Across all heading variants and datasets (48 points): Spearman between mean informativeness and mean gain = 0.199 (p = 0.17).


## D. Headroom from choosing the path weight per document or per question (MRR@10, in-sample)

| set | fixed alpha=0.2 | best fixed alpha | alpha=0 | per-document oracle | per-question oracle |
|---|---|---|---|---|---|
| GovReport (ICT) | 0.691 | 0.1 (0.696) | 0.693 | 0.733 | 0.754 |
| GovReport (LLM-written) | 0.878 | 0.1 (0.879) | 0.875 | 0.918 | 0.920 |
| QASPER (human) | 0.542 | 0.4 (0.546) | 0.515 | 0.616 | 0.639 |
| NQ (human) | 0.616 | 0.1 (0.624) | 0.624 | 0.696 | 0.696 |
| Pooled (3 datasets) | 0.615 | 0.1 (0.616) | 0.611 | 0.682 | 0.695 |


## E. Setting the path weight (2-fold cross-validation over documents, 5 repeats; test MRR@10)

| set | n | fixed alpha=0.2 | fixed, tuned | informativeness gate | query gate (ridge) |
|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 0.691 | 0.693 | 0.693 | 0.689 |
| GovReport (LLM-written) | 398 | 0.878 | 0.870 | 0.874 | 0.875 |
| QASPER (human) | 1256 | 0.542 | 0.543 | 0.540 | 0.542 |
| NQ (human) | 1744 | 0.616 | 0.622 | 0.623 | 0.623 |
| Pooled (3 datasets) | 4179 | 0.615 | 0.615 | 0.617 | 0.615 |



| set | comparison | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| GovReport (ICT) | fixed, tuned vs fixed alpha=0.2 | 0.002 | 0.435 | 0.999 |
| GovReport (ICT) | informativeness gate vs fixed alpha=0.2 | 0.002 | 0.402 | 0.865 |
| GovReport (ICT) | query gate (ridge) vs fixed alpha=0.2 | -0.002 | 0.490 | 0.254 |
| GovReport (LLM-written) | fixed, tuned vs fixed alpha=0.2 | -0.008 | 0.078 | 0.562 |
| GovReport (LLM-written) | informativeness gate vs fixed alpha=0.2 | -0.004 | 0.383 | 0.645 |
| GovReport (LLM-written) | query gate (ridge) vs fixed alpha=0.2 | -0.003 | 0.414 | 0.642 |
| QASPER (human) | fixed, tuned vs fixed alpha=0.2 | 0.001 | 0.735 | 0.860 |
| QASPER (human) | informativeness gate vs fixed alpha=0.2 | -0.002 | 0.437 | 0.276 |
| QASPER (human) | query gate (ridge) vs fixed alpha=0.2 | 0.000 | 0.929 | 0.217 |
| NQ (human) | fixed, tuned vs fixed alpha=0.2 | 0.006 | 0.041 | 0.830 |
| NQ (human) | informativeness gate vs fixed alpha=0.2 | 0.007 | 0.014 | 0.619 |
| NQ (human) | query gate (ridge) vs fixed alpha=0.2 | 0.007 | 0.012 | 0.552 |
| Pooled (3 datasets) | fixed, tuned vs fixed alpha=0.2 | -0.000 | 0.975 | 0.607 |
| Pooled (3 datasets) | informativeness gate vs fixed alpha=0.2 | 0.002 | 0.121 | 0.117 |
| Pooled (3 datasets) | query gate (ridge) vs fixed alpha=0.2 | -0.000 | 0.985 | 0.052 |
