# Ambient SFT experiment progress

Updated 2026-11-24 22:34 +08:00

- [x] Re-evaluate IFEval at 512 output tokens using the three existing checkpoints and compare against their 2,048-token results.
- [x] Prepare one fixed Dolci train and holdout split for the larger comparison.
- [x] Train SFT, T=1 distillation, and T=2.5 Ambient SFT for 2,048 steps each at seeds 42, 43, and 44.
- [x] Score all checkpoints on the full IFEval and GSM8K benchmarks, then summarize results across seeds.
- [x] Verify and publish the aggregate experiment results.

The larger comparison uses Qwen3-0.6B-Base, a 384-token training limit, one example per update, and the same linear learning-rate schedule across all conditions. Within each seed, the two teacher-based runs share the same branch draws. IFEval uses the paper's 2,048-token budget for the primary comparison and GSM8K uses eight fixed examples with a 512-token budget.

## Qwen3-1.7B full-parameter comparison

- [x] Benchmark the original Qwen3-1.7B-Base checkpoint on IFEval and GSM8K.
- [x] Verify the single-GPU full-parameter training path with BF16 weights, 8-bit AdamW state, and a frozen CPU teacher.
- [~] Train and evaluate SFT, T=1 distillation, and T=2.5 Ambient SFT for 2,048 updates each at seed 42.
- [ ] Verify and publish the full comparison and its relationship to the prior 0.6B results and the paper's baselines.

The three 1.7B runs reuse the pinned 4,000/256 Dolci split, 384-token limit, one example per update, and linear schedule from the completed 0.6B comparison. They use the same 8-bit optimizer and BF16 weights across all three conditions.

## Progress log

- 2026-11-24 20:51 +08:00 — Started the output-budget check and fixed-split preparation. The previous 512-step checkpoints and their full benchmark responses are available locally.
- 2026-11-24 20:55 +08:00 — Finished the output-budget check on all 541 prompts. The Ambient SFT margin over SFT was 1.85 percentage points at 512 tokens and 3.33 points at 2,048 tokens.
- 2026-11-24 20:55 +08:00 — The viewer rate-limited the larger fetch. Prepared the 4,000/256 split from a pinned Dolci Parquet shard and checked exact benchmark prompt overlap. Started the three-seed training queue.
- 2026-11-24 21:00 +08:00 — Published the benchmark runner and preparation commands. Verified the regenerated split hashes, all rendered benchmark prompts, and scorer output against the recorded 512-step run.
- 2026-11-24 21:02 +08:00 — Compared the two IFEval runs prompt by prompt and recorded how often the shorter output matched the longer output's prefix. Seed 42 SFT completed 2,048 steps and produced its checkpoint.
- 2026-11-24 21:19 +08:00 — Completed training and full benchmark scoring for all three methods at seed 42. Started seed 43 and recorded the first seed in the expanded results.
- 2026-11-24 21:44 +08:00 — Completed seed 43 and added it to the benchmark comparison. Started the third and final seed.
- 2026-11-24 22:00 +08:00 — All nine training checkpoints are saved. Started the final seed's benchmark generation.
- 2026-11-24 22:08 +08:00 — Finished and verified all nine full benchmark runs. The final report includes mean scores, per-seed deltas, matching branch schedules, and checkpoints verified by SHA-256.
- 2026-11-24 22:29 +08:00 — Started the independent 1.7B base evaluation and prepared the full-parameter single-GPU run.
- 2026-11-24 22:34 +08:00 — The 1.7B base scored 34.38% on IFEval and 52.24% on GSM8K. The full-parameter runner passed both branches and held-out evaluation in a 12-step CUDA test. Started the 2,048-step comparison.
