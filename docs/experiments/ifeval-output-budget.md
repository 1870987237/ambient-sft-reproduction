# IFEval output budget on the 512-step checkpoints

The three Qwen3-0.6B-Base checkpoints were trained with seed 42. This comparison uses all 541 official IFEval prompts, the same OLMo 3 chat template and greedy vLLM FlashAttention decoding. Each row is prompt-level loose accuracy using the verifier in [the paper's pinned OLMES revision](https://github.com/allenai/olmes/commit/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e). The 512-token budget was rerun and the 2,048-token responses were rescored.

| Method | 512-token cap | 2,048-token cap | Responses at the cap, 512 / 2,048 |
| --- | ---: | ---: | ---: |
| SFT | 12.94% (70/541) | 12.94% (70/541) | 42 / 42 |
| T=1 distillation | 14.23% (77/541) | 14.60% (79/541) | 105 / 104 |
| Ambient SFT, T=2.5 | 14.79% (80/541) | 16.27% (88/541) | 217 / 214 |

Ambient SFT passed 10 more prompts than SFT at 512 tokens and 18 more at 2,048 tokens. The paired 95% item-bootstrap interval for its margin over SFT at 512 tokens was −1.11 to +4.81 percentage points. The margin was smaller at the shorter budget in this checkpoint comparison. Estimates use one training seed.

The 512-token response was an exact text prefix of the 2,048-token response on 540 of 541 prompts for SFT and T=1 distillation, and on 392 of 541 for Ambient SFT. Of the eight net prompts Ambient SFT lost at the shorter budget, four came from same-prefix responses and four came from responses with different text prefixes. The comparison measures separate generation runs at each budget.

All three checkpoint outputs and ground-truth IDs were matched before scoring. Full aggregate counts and paired differences are in [the result JSON](ifeval-output-budget.json).
