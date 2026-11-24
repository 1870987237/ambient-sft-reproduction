# 2,048-step Ambient SFT comparison

Completed seeds **42** (1 of 3 planned).

Each run starts from Qwen3-0.6B-Base and uses the same 4,000 Dolci training candidates, of which 3,390 still contain supervised tokens at the 384-token limit. Training applies 2,048 updates, one example per update, with the same linear schedule within each seed. The two distillation methods use the same micro-batch branches at each seed. [Benchmark and training protocol](local-benchmark-protocol.md)

## Full benchmark scores

IFEval uses prompt-level loose accuracy on 541 prompts. GSM8K uses 8-shot exact match on 1,319 test problems.

| Seed | Method | IFEval | GSM8K | IFEval responses at 2,048-token cap |
| --- | --- | ---: | ---: | ---: |
| 42 | SFT | 24.40% (132/541) | 4.32% (57/1,319) | 101 |
| 42 | T=1 distillation | 22.00% (119/541) | 4.32% (57/1,319) | 111 |
| 42 | Ambient SFT, T=2.5 | 18.48% (100/541) | 4.85% (64/1,319) | 94 |

## Differences within the same seed

Differences are absolute percentage points. Seed-specific values appear in the order listed above.

| Comparison | IFEval mean (by seed) | GSM8K mean (by seed) |
| --- | ---: | ---: |
| T=1 distillation vs. SFT | -2.40 (-2.40) | +0.00 (+0.00) |
| Ambient SFT vs. SFT | -5.91 (-5.91) | +0.53 (+0.53) |
| Ambient SFT vs. T=1 distillation | -3.51 (-3.51) | +0.53 (+0.53) |

[Verified results JSON](ambient-sft-2048.json) contains training summaries, counts, per-seed paired item
intervals and split hashes. The per-seed intervals quantify test-item variation conditional on each checkpoint.
