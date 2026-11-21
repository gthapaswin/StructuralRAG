# Held-out test: ConditionalQA results against the pre-registered predictions

Predictions: [`heldout_predictions.md`](heldout_predictions.md) (committed and pushed before this run). n = 2422 questions; tests are document-clustered sign-flip tests; 95% bootstrap CIs.

MRR@10: flat 0.700, structure-aware 0.695, placebo 0.692, role-neutral 0.706.

| | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | SA beats flat (+0.02 to +0.03) | -0.005 [-0.015, +0.005], p = 0.386 | **fails** |
| P2 | Real beats shuffled headings | +0.003 [-0.005, +0.011], p = 0.451 | **fails** |
| P3 | Gain rises with target specificity | Spearman 0.032, p = 0.118 | **fails** |
| P4 | Summary-only targets lose | mean gain -0.207 (n = 74) | **holds** |
| P5 | Unique > generic targets | unique -0.004 (n = 1705) vs generic -0.082 (n = 216) | **holds** |
| P6 | Role-neutral ≥ SA − 0.005 | +0.011 [+0.007, +0.014], p = 0.000 | **holds** |
| P7 | Role-neutral > SA on summary-only targets | +0.125 (n = 74), p = 0.000 | **holds** |
