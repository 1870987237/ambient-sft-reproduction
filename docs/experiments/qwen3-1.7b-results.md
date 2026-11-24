# Qwen3-1.7B-Base full-parameter comparison

Completed conditions **base, sft, kd, ambient** (4 of 4).

The three trained checkpoints use seed 42, 2,048 updates, the same 4,000 Dolci candidates, one example per update, a 384-token limit, and 8-bit AdamW. IFEval uses 541 prompts at a 2,048-token cap. GSM8K uses all 1,319 test problems with the eight fixed OLMES examples. [Training and evaluation protocol](qwen3-1.7b-protocol.md)

| Condition | IFEval loose | GSM8K exact match | Peak allocated VRAM during training |
| --- | ---: | ---: | ---: |
| Original base | 34.38% (186/541) | 52.24% (689/1,319) | — |
| SFT | 47.32% (256/541) | 52.31% (690/1,319) | 11.09 GiB |
| T=1 distillation | 38.26% (207/541) | 57.39% (757/1,319) | 11.29 GiB |
| Ambient SFT, T=2.5 | 35.49% (192/541) | 56.94% (751/1,319) | 11.29 GiB |

| Comparison | IFEval change (percentage points) | GSM8K change (percentage points) |
| --- | ---: | ---: |
| SFT vs. Original base | +12.94 | +0.08 |
| T=1 distillation vs. Original base | +3.88 | +5.16 |
| Ambient SFT, T=2.5 vs. Original base | +1.11 | +4.70 |
| T=1 distillation vs. SFT | -9.06 | +5.08 |
| Ambient SFT, T=2.5 vs. SFT | -11.83 | +4.62 |
| Ambient SFT, T=2.5 vs. T=1 distillation | -2.77 | -0.45 |

Ambient SFT answered 61 more GSM8K problems correctly and passed 64 fewer IFEval prompts than SFT. T=1 distillation scored higher than Ambient SFT on both tasks, by six GSM8K problems and 15 IFEval prompts. These comparisons use one training seed. The original base GSM8K score was 1.26 percentage points below the paper's base score, and local SFT added one correct GSM8K answer.

The local 0.6B seed-42 run improved GSM8K by 0.53 percentage points over SFT. The corresponding improvement at 1.7B is 4.62 points. Model size, parameter precision, and optimizer state precision changed between the two local experiments. [0.6B results](ambient-sft-2048.md)

In the paper's Qwen3-1.7B experiment, GSM8K was 53.5% for base, 61.0% for SFT, and 72.3% for Ambient SFT. Its IFEval scores were 37.9%, 58.8%, and 52.3%. The local results use a smaller training budget and an 8-bit optimizer. [Paper Table 2](https://arxiv.org/html/2611.23229v1#S4.SS3)

[Verified results JSON](qwen3-1.7b-results.json) contains counts, checkpoint hashes, and paired item intervals.
