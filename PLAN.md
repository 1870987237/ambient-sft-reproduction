# Ambient SFT experiment progress

Updated 2026-11-24 21:44 +08:00

- [x] Re-evaluate IFEval at 512 output tokens using the three existing checkpoints and compare against their 2,048-token results.
- [x] Prepare one fixed Dolci train and holdout split for the larger comparison.
- [~] Train SFT, T=1 distillation, and T=2.5 Ambient SFT for 2,048 steps each at seeds 42, 43, and 44. Seeds 42 and 43 are complete and seed 44 is training.
- [ ] Score all checkpoints on the full IFEval and GSM8K benchmarks, then summarize results across seeds.
- [ ] Verify and publish the aggregate experiment results.

The larger comparison uses Qwen3-0.6B-Base, a 384-token training limit, one example per update, and the same linear learning-rate schedule across all conditions. Within each seed, the two teacher-based runs share the same branch draws. IFEval uses the paper's 2,048-token budget for the primary comparison and GSM8K uses eight fixed examples with a 512-token budget.

## Progress log

- 2026-11-24 20:51 +08:00 — Started the output-budget check and fixed-split preparation. The previous 512-step checkpoints and their full benchmark responses are available locally.
- 2026-11-24 20:55 +08:00 — Finished the output-budget check on all 541 prompts. The Ambient SFT margin over SFT was 1.85 percentage points at 512 tokens and 3.33 points at 2,048 tokens.
- 2026-11-24 20:55 +08:00 — The viewer rate-limited the larger fetch. Prepared the 4,000/256 split from a pinned Dolci Parquet shard and checked exact benchmark prompt overlap. Started the three-seed training queue.
- 2026-11-24 21:00 +08:00 — Published the benchmark runner and preparation commands. Verified the regenerated split hashes, all rendered benchmark prompts, and scorer output against the recorded 512-step run.
- 2026-11-24 21:02 +08:00 — Compared the two IFEval runs prompt by prompt and recorded how often the shorter output matched the longer output's prefix. Seed 42 SFT completed 2,048 steps and produced its checkpoint.
- 2026-11-24 21:19 +08:00 — Completed training and full benchmark scoring for all three methods at seed 42. Started seed 43 and recorded the first seed in the expanded results.
- 2026-11-24 21:44 +08:00 — Completed seed 43 and added it to the benchmark comparison. Started the third and final seed.
