# 2,048-step Ambient SFT comparison

Completed seeds **42, 43, 44** (3 of 3 planned).

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
| 44 | SFT | 25.32% (137/541) | 6.90% (91/1,319) | 153 |
| 44 | T=1 distillation | 20.52% (111/541) | 5.84% (77/1,319) | 120 |
| 44 | Ambient SFT, T=2.5 | 21.63% (117/541) | 6.60% (87/1,319) | 142 |

Scores averaged across completed seeds, with sample standard deviation in percentage points.

| Method | IFEval mean ± SD | GSM8K mean ± SD |
| --- | ---: | ---: |
| SFT | 24.46% ± 0.83 | 5.56% ± 1.29 |
| T=1 distillation | 22.18% ± 1.76 | 4.80% ± 0.90 |
| Ambient SFT, T=2.5 | 21.01% ± 2.28 | 6.09% ± 1.08 |

## Differences within the same seed

Differences are absolute percentage points. Seed-specific values appear in the order listed above.

| Comparison | IFEval mean (by seed) | GSM8K mean (by seed) |
| --- | ---: | ---: |
| T=1 distillation vs. SFT | -2.28 (-2.40, +0.37, -4.81) | -0.76 (+0.00, -1.21, -1.06) |
| Ambient SFT vs. SFT | -3.45 (-5.91, -0.74, -3.70) | +0.53 (+0.53, +1.36, -0.30) |
| Ambient SFT vs. T=1 distillation | -1.17 (-3.51, -1.11, +1.11) | +1.29 (+0.53, +2.58, +0.76) |

Ambient SFT scored lower than SFT on IFEval in all three seeds. On GSM8K it improved over T=1 distillation in all three seeds and over SFT in two of three seeds. Its mean GSM8K gain over SFT was 0.53 percentage points. The paper reports an 11.3-point GSM8K gain for Qwen3-1.7B in its 1%-data experiment. [Paper Table 2](https://arxiv.org/html/2611.23229v1#S4.SS3)

[Verified results JSON](ambient-sft-2048.json) contains training summaries, counts, per-seed paired item
intervals and split hashes. The per-seed intervals quantify test-item variation conditional on each checkpoint.
