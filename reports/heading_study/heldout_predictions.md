# Held-out test: predictions written before running retrieval (ConditionalQA)

**Status: APPROVED by the author on 2026-11-21, before any ConditionalQA retrieval was run.**
Once approved, this file is committed and pushed before any ConditionalQA retrieval result is
computed; the commit timestamp is the record. Results are reported against these predictions
whether they hold or not.

## The collection

ConditionalQA (Sun et al., ACL 2022): 495 UK government guidance pages (gov.uk), 9,566 sections
(up to 4 levels: page parts such as "Eligibility" and "How to claim", with nested sections), and
2,422 answerable questions written by crowd workers, from the train and dev splits (no model is
trained on them). Gold sections = sections containing the question's marked evidence; 49% of
questions have more than one gold section, and a question counts as answered if any gold section
is ranked.

## What was looked at before writing this

Only the pages' headings and the questions' gold sections, never any retrieval score. Target
heading specificity w = log(N/df)/log N (max over a question's gold sections):

| Target heading | share of questions | pooled gain in the four existing question sets |
|---|---|---|
| unique (w ≥ 0.9) | 70.4% | +0.034 |
| shared (0.5 ≤ w < 0.9) | 20.7% | +0.017 |
| generic (w < 0.5) | 8.9% | −0.045 |

Questions whose gold sections are all summary sections ("Overview", "Introduction", …): 3.1%.
The most common generic target headings are "Overview" (505 gold sections), "Eligibility" (140),
"What you'll get" (51) and "How to apply" (31): like QASPER's "Results", most are functional page
parts that questions target directly.

## Fixed decisions (not changed after seeing results)

- Within-document section retrieval, MRR@10. Query = the question only (the scenario is not used).
- Methods and settings exactly as in the other domains, from GovReport dev: flat = RRF of BM25 and
  dense over section text; structure-aware (SA) = path weight alpha = 0.2, beta = 0, BM25 over the
  structural view, RRF k = 60; encoder bge-small-en-v1.5. Placebo = three shuffles (seeds 0-2).
- Tests: document-clustered sign-flip test; a prediction "holds" if the effect has the predicted
  sign and p < 0.05 (P1-P3), or the predicted sign (P4-P5, small or descriptive).

## Predictions

| | Prediction | Basis |
|---|---|---|
| P1 | SA beats flat: mean gain > 0, expected about +0.02 to +0.03 MRR@10 | target-heading mix × pooled per-category gains (+0.023 all sets, +0.034 human-question sets) |
| P2 | Real headings beat shuffled headings (placebo) | headings carry meaning when targets are mostly specific |
| P3 | Per-question gain rises with target heading specificity (Spearman > 0) | the specificity account (all four existing sets) |
| P4 | Questions whose gold sections are all summary sections lose (mean gain < 0) | summary-section account (n ≈ 75: direction only) |
| P5 | Gain for unique target headings > gain for generic ones | specificity account |

What would count against the account: P1 or P3 failing (no gain, or no relation to specificity),
or P5 reversing.

## The role-aware method (frozen before this test)

"Summary-role neutral": sections whose heading marks a summary role (Introduction, Abstract,
Overview, Summary, Conclusion(s), Concluding, Background; fixed regular expression in
`src/structuralrag/analysis/headings.py`) get a neutral path score (the document's mean path
similarity); all other headings are used as in SA. Same alpha = 0.2, no other parameters.
Development results (the four existing question sets, used to design it; not a clean test):
GovReport ICT +0.001, GovReport LLM-written +0.008, QASPER −0.005, NQ +0.011 vs SA.

| | Prediction | Basis |
|---|---|---|
| P6 | Not worse than SA overall: role-neutral ≥ SA − 0.005 MRR@10 | development results; only 3.1% of questions target summary sections alone |
| P7 | On questions whose gold sections are all summary sections, role-neutral > SA | it removes the penalty on generic summary headings |

## Not tested here

A question-side "does this question want a summary?" classifier was dropped before this test: on
the development sets it was weak (cross-validated AUC 0.63 on NQ and QASPER, 0.51 on GovReport
LLM-written questions).
