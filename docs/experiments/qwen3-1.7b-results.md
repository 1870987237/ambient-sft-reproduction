# Qwen3-1.7B-Base full-parameter comparison

Completed conditions **base, sft** (2 of 4).

The three trained checkpoints use seed 42, 2,048 updates, the same 4,000 Dolci candidates, one example per update, a 384-token limit, and 8-bit AdamW. IFEval uses 541 prompts at a 2,048-token cap. GSM8K uses all 1,319 test problems with the eight fixed OLMES examples. [Training and evaluation protocol](qwen3-1.7b-protocol.md)

| Condition | IFEval loose | GSM8K exact match | Peak allocated VRAM during training |
| --- | ---: | ---: | ---: |
| Original base | 34.38% (186/541) | 52.24% (689/1,319) | — |
| SFT | 47.32% (256/541) | 52.31% (690/1,319) | 11.09 GiB |

| Comparison | IFEval change (percentage points) | GSM8K change (percentage points) |
| --- | ---: | ---: |
| SFT vs. Original base | +12.94 | +0.08 |

In the paper's Qwen3-1.7B experiment, GSM8K was 53.5% for base, 61.0% for SFT, and 72.3% for Ambient SFT. Its IFEval scores were 37.9%, 58.8%, and 52.3%. The local results use a smaller training budget and an 8-bit optimizer. [Paper Table 2](https://arxiv.org/html/2611.23229v1#S4.SS3)

[Verified results JSON](qwen3-1.7b-results.json) contains counts, checkpoint hashes, and paired item intervals.
