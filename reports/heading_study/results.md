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
| GovReport (ICT) | 1179 | 0.072 | 0.011 | 0.133 | 0.013 |
| GovReport (LLM-written) | 398 | -0.080 | -0.180 | 0.025 | 0.113 |
| QASPER (human) | 1256 | 0.043 | -0.015 | 0.096 | 0.132 |
| NQ (human) | 1744 | 0.099 | 0.054 | 0.144 | 0.000 |


Across all heading variants and datasets (48 points): Spearman between mean informativeness and mean gain = 0.199 (p = 0.17).


## D. Headroom from choosing the path weight per document or per question (MRR@10, in-sample)

| set | fixed alpha=0.2 | best fixed alpha | alpha=0 | per-document oracle | per-question oracle |
|---|---|---|---|---|---|
| GovReport (ICT) | 0.691 | 0.1 (0.696) | 0.693 | 0.734 | 0.754 |
| GovReport (LLM-written) | 0.878 | 0.1 (0.879) | 0.875 | 0.918 | 0.920 |
| QASPER (human) | 0.542 | 0.4 (0.546) | 0.515 | 0.618 | 0.639 |
| NQ (human) | 0.616 | 0.1 (0.624) | 0.624 | 0.696 | 0.696 |
| Pooled (3 datasets) | 0.615 | 0.1 (0.616) | 0.611 | 0.683 | 0.695 |


## E. Setting the path weight (2-fold cross-validation over documents, 5 repeats; test MRR@10)

| set | n | fixed alpha=0.2 | fixed, tuned | informativeness gate | query gate (ridge) |
|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 0.691 | 0.695 | 0.691 | 0.691 |
| GovReport (LLM-written) | 398 | 0.878 | 0.874 | 0.876 | 0.874 |
| QASPER (human) | 1256 | 0.542 | 0.542 | 0.547 | 0.542 |
| NQ (human) | 1744 | 0.616 | 0.623 | 0.623 | 0.624 |
| Pooled (3 datasets) | 4179 | 0.615 | 0.614 | 0.615 | 0.615 |



| set | comparison | mean_diff | p_permutation | p_wilcoxon |
|---|---|---|---|---|
| GovReport (ICT) | fixed, tuned vs fixed alpha=0.2 | 0.004 | 0.124 | 0.759 |
| GovReport (ICT) | informativeness gate vs fixed alpha=0.2 | -0.000 | 0.956 | 0.997 |
| GovReport (ICT) | query gate (ridge) vs fixed alpha=0.2 | -0.000 | 0.988 | 0.931 |
| GovReport (LLM-written) | fixed, tuned vs fixed alpha=0.2 | -0.004 | 0.233 | 0.826 |
| GovReport (LLM-written) | informativeness gate vs fixed alpha=0.2 | -0.002 | 0.662 | 0.988 |
| GovReport (LLM-written) | query gate (ridge) vs fixed alpha=0.2 | -0.004 | 0.280 | 0.766 |
| QASPER (human) | fixed, tuned vs fixed alpha=0.2 | 0.000 | 0.972 | 0.418 |
| QASPER (human) | informativeness gate vs fixed alpha=0.2 | 0.005 | 0.223 | 0.934 |
| QASPER (human) | query gate (ridge) vs fixed alpha=0.2 | 0.000 | 0.980 | 0.424 |
| NQ (human) | fixed, tuned vs fixed alpha=0.2 | 0.007 | 0.024 | 0.624 |
| NQ (human) | informativeness gate vs fixed alpha=0.2 | 0.007 | 0.021 | 0.679 |
| NQ (human) | query gate (ridge) vs fixed alpha=0.2 | 0.007 | 0.007 | 0.655 |
| Pooled (3 datasets) | fixed, tuned vs fixed alpha=0.2 | -0.001 | 0.613 | 0.460 |
| Pooled (3 datasets) | informativeness gate vs fixed alpha=0.2 | 0.000 | 0.948 | 0.195 |
| Pooled (3 datasets) | query gate (ridge) vs fixed alpha=0.2 | -0.000 | 0.876 | 0.045 |


## F. Fine-tuning absorption (gain of SA by informativeness tercile)

| set | encoder | informativeness | n | gain_vs_flat | gain_vs_placebo |
|---|---|---|---|---|---|
| GovReport (ICT) | base | low | 393 | -0.014 | -0.010 |
| GovReport (ICT) | base | mid | 393 | 0.015 | -0.004 |
| GovReport (ICT) | base | high | 393 | 0.003 | -0.011 |
| GovReport (ICT) | fine-tuned | low | 393 | -0.017 | 0.001 |
| GovReport (ICT) | fine-tuned | mid | 393 | 0.008 | 0.003 |
| GovReport (ICT) | fine-tuned | high | 393 | 0.003 | -0.001 |
| GovReport (LLM-written) | base | low | 133 | 0.034 | 0.026 |
| GovReport (LLM-written) | base | mid | 132 | 0.004 | 0.023 |
| GovReport (LLM-written) | base | high | 133 | 0.005 | -0.002 |
| GovReport (LLM-written) | fine-tuned | low | 133 | 0.012 | 0.007 |
| GovReport (LLM-written) | fine-tuned | mid | 132 | 0.007 | 0.010 |
| GovReport (LLM-written) | fine-tuned | high | 133 | 0.021 | 0.008 |
| QASPER (human) | base | low | 419 | 0.034 | 0.034 |
| QASPER (human) | base | mid | 418 | 0.030 | 0.028 |
| QASPER (human) | base | high | 419 | 0.034 | 0.031 |
| QASPER (human) | fine-tuned | low | 419 | 0.035 | 0.019 |
| QASPER (human) | fine-tuned | mid | 418 | 0.013 | 0.017 |
| QASPER (human) | fine-tuned | high | 419 | 0.036 | 0.034 |
| NQ (human) | base | low | 582 | -0.019 | 0.005 |
| NQ (human) | base | mid | 581 | -0.022 | -0.005 |
| NQ (human) | base | high | 581 | -0.029 | -0.028 |
| NQ (human) | fine-tuned | low | 582 | -0.017 | 0.002 |
| NQ (human) | fine-tuned | mid | 581 | -0.023 | -0.011 |
| NQ (human) | fine-tuned | high | 581 | -0.020 | -0.019 |


## H. Gain by the target section's heading specificity (w = log(N/df)/log N)

The last row per set is the Spearman correlation between w and the per-question gain (p in last column).

| set | gold heading | n | gain_vs_flat | ci_lo | ci_hi |
|---|---|---|---|---|---|
| GovReport (ICT) | generic (w < 0.5) | 175 | -0.076 | -0.107 | -0.046 |
| GovReport (ICT) | shared (0.5-0.9) | 49 | -0.066 | -0.126 | -0.009 |
| GovReport (ICT) | unique (w >= 0.9) | 955 | 0.019 | 0.008 | 0.031 |
| GovReport (ICT) | Spearman 0.224 [0.165, 0.286] | 1179 | nan | nan | 0.000 |
| GovReport (LLM-written) | generic (w < 0.5) | 65 | -0.103 | -0.160 | -0.050 |
| GovReport (LLM-written) | shared (0.5-0.9) | 19 | 0.009 | -0.070 | 0.088 |
| GovReport (LLM-written) | unique (w >= 0.9) | 314 | 0.039 | 0.022 | 0.057 |
| GovReport (LLM-written) | Spearman 0.309 [0.213, 0.400] | 398 | nan | nan | 0.000 |
| QASPER (human) | generic (w < 0.5) | 477 | 0.003 | -0.015 | 0.020 |
| QASPER (human) | shared (0.5-0.9) | 274 | 0.039 | 0.013 | 0.065 |
| QASPER (human) | unique (w >= 0.9) | 505 | 0.057 | 0.038 | 0.077 |
| QASPER (human) | Spearman 0.184 [0.131, 0.236] | 1256 | nan | nan | 0.000 |
| NQ (human) | generic (w < 0.5) | 1079 | -0.054 | -0.066 | -0.042 |
| NQ (human) | shared (0.5-0.9) | 232 | 0.010 | -0.017 | 0.038 |
| NQ (human) | unique (w >= 0.9) | 433 | 0.036 | 0.016 | 0.054 |
| NQ (human) | Spearman 0.155 [0.106, 0.200] | 1744 | nan | nan | 0.000 |


## I. Specificity fallback (alpha tuned on GovReport dev: SA 0.2, neutral 0.2, content 0.2; MRR@10)

| set | n | flat | SA (a=0.2) | + fallback, neutral (a=0.2) | + fallback, content (a=0.2) |
|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 0.689 | 0.691 | 0.693 | 0.694 |
| GovReport (LLM-written) | 398 | 0.864 | 0.878 | 0.887 | 0.890 |
| QASPER (human) | 1256 | 0.509 | 0.542 | 0.530 | 0.514 |
| NQ (human) | 1744 | 0.640 | 0.616 | 0.624 | 0.628 |



| set | comparison | mean_diff | p_permutation | p_holm |
|---|---|---|---|---|
| GovReport (ICT) | fallback (neutral) vs SA | 0.002 | 0.207 | 0.828 |
| GovReport (ICT) | fallback (content) vs SA | 0.003 | 0.322 | 0.966 |
| GovReport (ICT) | fallback (neutral) vs flat | 0.004 | 0.426 | 0.966 |
| GovReport (ICT) | SA vs flat | 0.002 | 0.767 | 0.966 |
| GovReport (LLM-written) | fallback (neutral) vs SA | 0.009 | 0.033 | 0.067 |
| GovReport (LLM-written) | fallback (content) vs SA | 0.012 | 0.018 | 0.054 |
| GovReport (LLM-written) | fallback (neutral) vs flat | 0.023 | 0.004 | 0.016 |
| GovReport (LLM-written) | SA vs flat | 0.014 | 0.123 | 0.123 |
| QASPER (human) | fallback (neutral) vs SA | -0.012 | 0.000 | 0.001 |
| QASPER (human) | fallback (content) vs SA | -0.027 | 0.000 | 0.000 |
| QASPER (human) | fallback (neutral) vs flat | 0.021 | 0.000 | 0.001 |
| QASPER (human) | SA vs flat | 0.033 | 0.000 | 0.000 |
| NQ (human) | fallback (neutral) vs SA | 0.008 | 0.006 | 0.012 |
| NQ (human) | fallback (content) vs SA | 0.012 | 0.008 | 0.012 |
| NQ (human) | fallback (neutral) vs flat | -0.016 | 0.001 | 0.002 |
| NQ (human) | SA vs flat | -0.023 | 0.000 | 0.000 |


By the target section's heading specificity:

| set | gold heading | n | SA - flat | neutral fallback - flat | content fallback - flat |
|---|---|---|---|---|---|
| GovReport (ICT) | generic (w < 0.5) | 175 | -0.076 | -0.047 | -0.005 |
| GovReport (ICT) | shared (0.5-0.9) | 49 | -0.066 | -0.071 | -0.069 |
| GovReport (ICT) | unique (w >= 0.9) | 955 | 0.019 | 0.017 | 0.010 |
| GovReport (LLM-written) | generic (w < 0.5) | 65 | -0.103 | -0.053 | -0.014 |
| GovReport (LLM-written) | shared (0.5-0.9) | 19 | 0.009 | 0.035 | 0.035 |
| GovReport (LLM-written) | unique (w >= 0.9) | 314 | 0.039 | 0.038 | 0.035 |
| QASPER (human) | generic (w < 0.5) | 477 | 0.003 | -0.000 | 0.013 |
| QASPER (human) | shared (0.5-0.9) | 274 | 0.039 | 0.015 | -0.001 |
| QASPER (human) | unique (w >= 0.9) | 505 | 0.057 | 0.044 | 0.001 |
| NQ (human) | generic (w < 0.5) | 1079 | -0.054 | -0.038 | -0.005 |
| NQ (human) | shared (0.5-0.9) | 232 | 0.010 | -0.001 | -0.025 |
| NQ (human) | unique (w >= 0.9) | 433 | 0.036 | 0.033 | -0.021 |
