# Heading study: effect on final answers

Fixed random sample per dataset (questions with a reference answer; ConditionalQA yes/no answers excluded). Top-3 sections → qwen3:8b cited answer (plain context, 900 words, no claim check). Correct = the answer contains a reference answer string. Document-clustered tests vs structure-aware.

| dataset | system | n | gold section in top-3 | answer correct | abstained |
|---|---|---|---|---|---|
| nq | flat | 400 | 0.750 | 0.613 | 0.098 |
| nq | structure-aware | 400 | 0.738 | 0.615 | 0.102 |
| nq | role-neutral | 400 | 0.748 | 0.618 | 0.100 |
| conditionalqa | flat | 400 | 0.777 | 0.487 | 0.198 |
| conditionalqa | structure-aware | 400 | 0.795 | 0.500 | 0.203 |
| conditionalqa | role-neutral | 400 | 0.800 | 0.512 | 0.198 |

| dataset | comparison (answer correct) | difference | 95% CI | p (clustered) |
|---|---|---|---|---|
| nq | role-neutral vs structure-aware | +0.003 | [-0.010, +0.015] | 1.000 |
| nq | flat vs structure-aware | -0.003 | [-0.030, +0.025] | 1.000 |
| nq | role-neutral vs flat | +0.005 | [-0.020, +0.030] | 0.848 |
| conditionalqa | role-neutral vs structure-aware | +0.013 | [+0.000, +0.028] | 0.123 |
| conditionalqa | flat vs structure-aware | -0.013 | [-0.052, +0.028] | 0.642 |
| conditionalqa | role-neutral vs flat | +0.025 | [-0.010, +0.062] | 0.289 |
