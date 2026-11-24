# 2,048-step Ambient SFT comparison

Completed seeds **42, 43** (2 of 3 planned).

Each run starts from Qwen3-0.6B-Base and uses the same 4,000 Dolci training candidates, of which 3,390 still contain supervised tokens at the 384-token limit. Training applies 2,048 updates, one example per update, with the same linear schedule within each seed. The two distillation methods use the same micro-batch branches at each seed. [Benchmark and training protocol](local-benchmark-protocol.md)

## Full benchmark scores

IFEval uses prompt-level loose accuracy on 541 prompts. GSM8K uses 8-shot exact match on 1,319 test problems.

| Seed | Method | IFEval | GSM8K | IFEval responses at 2,048-token cap |
| --- | --- | ---: | ---: | ---: |
| 42 | SFT | 24.40% (132/541) | 4.32% (57/1,319) | 101 |
| 42 | T=1 distillation | 22.00% (119/541) | 4.32% (57/1,319) | 111 |
| 42 | Ambient SFT, T=2.5 | 18.48% (100/541) | 4.85% (64/1,319) | 94 |
| 43 | SFT | 23.66% (128/541) | 5.46% (72/1,319) | 172 |
| 43 | T=1 distillation | 24.03% (130/541) | 4.25% (56/1,319) | 127 |
| 43 | Ambient SFT, T=2.5 | 22.92% (124/541) | 6.82% (90/1,319) | 136 |

Scores averaged across completed seeds, with sample standard deviation in percentage points.

| Method | IFEval mean ± SD | GSM8K mean ± SD |
| --- | ---: | ---: |
| SFT | 24.03% ± 0.52 | 4.89% ± 0.80 |
| T=1 distillation | 23.01% ± 1.44 | 4.28% ± 0.05 |
| Ambient SFT, T=2.5 | 20.70% ± 3.14 | 5.84% ± 1.39 |

## Differences within the same seed

Differences are absolute percentage points. Seed-specific values appear in the order listed above.

| Comparison | IFEval mean (by seed) | GSM8K mean (by seed) |
| --- | ---: | ---: |
| T=1 distillation vs. SFT | -1.02 (-2.40, +0.37) | -0.61 (+0.00, -1.21) |
| Ambient SFT vs. SFT | -3.33 (-5.91, -0.74) | +0.95 (+0.53, +1.36) |
| Ambient SFT vs. T=1 distillation | -2.31 (-3.51, -1.11) | +1.55 (+0.53, +2.58) |

[Verified results JSON](ambient-sft-2048.json) contains training summaries, counts, per-seed paired item
intervals and split hashes. The per-seed intervals quantify test-item variation conditional on each checkpoint.
