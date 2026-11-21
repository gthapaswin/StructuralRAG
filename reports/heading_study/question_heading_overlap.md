# Heading study: do questions use their target heading's words?

Target = first gold section. 'Heading word in question' = at least one content word of the target's own heading occurs in the question. 'Heading points to target' = the target's heading path is the closest in its document to the question (bge-small cosine). Gain = structure-aware minus flat (MRR@10, alpha 0.2).

| set | n | heading word in question | heading points to target | SA gain overall | gain: with / without heading word | gain: points / does not | Spearman(lexical, gain) | same, unique targets only |
|---|---|---|---|---|---|---|---|---|
| GovReport (ICT) | 1179 | 58% | 28% | +0.002 | +0.027 / -0.034 | +0.043 / -0.015 | 0.18 (p = 3e-10) | 0.12 (p = 0.00019) |
| GovReport (LLM-written) | 398 | 75% | 48% | +0.014 | +0.044 / -0.076 | +0.040 / -0.009 | 0.22 (p = 1.3e-05) | 0.01 (p = 0.85) |
| QASPER (human) | 1256 | 19% | 29% | +0.033 | +0.112 / +0.014 | +0.106 / +0.002 | 0.19 (p = 5e-12) | 0.19 (p = 2.4e-05) |
| NQ (human) | 1744 | 10% | 19% | -0.026 | +0.068 / -0.036 | +0.045 / -0.042 | 0.15 (p = 1.1e-09) | 0.11 (p = 0.03) |
| ConditionalQA (human) | 2422 | 38% | 24% | -0.005 | +0.057 / -0.043 | +0.091 / -0.036 | 0.19 (p = 5.9e-21) | 0.14 (p = 3.9e-07) |
