# Ambient SFT reproduction

An independent single-device implementation of [Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching](https://arxiv.org/abs/2611.23229v1) by Daras et al. It compares vanilla SFT with Ambient SFT on the same instruction data, model initialization, training order, and held-out examples.

The runner implements Algorithm 1 with one example per micro-batch and one micro-batch per optimizer step. Each micro-batch uses hard-label cross-entropy with probability `0.75`. Otherwise, the frozen base model and student receive the same input, both sets of logits are divided by `2.5`, and the student minimizes full-vocabulary soft-label cross-entropy. Both branches use the same assistant-only next-token mask, including the end-of-turn token. The temperature matching loss uses no additional temperature-squared multiplier. Use `--temperature 1` for the untempered distillation control, or `--mode sft` for vanilla SFT.

## Run the local comparison

Use a CUDA PyTorch build that supports your GPU and bfloat16 if running on CUDA. See the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the appropriate wheel. The full-parameter runner keeps FP32 student parameters and AdamW state, with BF16 CUDA autocast and a BF16 frozen teacher. A CUDA machine with 16 GB of VRAM is suitable for the default 0.6B model and short sequences. `--device cpu` is available for small local checks.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/prepare_dolci.py
python scripts/train.py --mode sft --train-data data/dolci/train.jsonl --eval-data data/dolci/eval.jsonl
python scripts/train.py --mode ambient --train-data data/dolci/train.jsonl --eval-data data/dolci/eval.jsonl
```

The default model is the pinned [Qwen3-0.6B-Base checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd). `--model` accepts a local checkpoint directory or another ChatML-compatible checkpoint. `--save-dir checkpoints/run-name` optionally saves the trained student and its tokenizer to a new directory. For a third comparison, repeat the Ambient command with `--temperature 1`.

The data script reads a bounded slice of the public [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) viewer, filters out tool-call rows and duplicates, and writes separate local training and evaluation JSONL files. The viewer can change over time. The script prints file SHA-256 hashes, and each run logs those hashes, usable row counts after truncation, and the number of supervised tokens in its held-out evaluation. Keep the generated files locally to repeat a specific run. Custom JSONL data uses one `messages` array per row with text `system`, `user`, and `assistant` turns.

## Scope

This first experiment checks the training path and reports token-weighted held-out next-token NLL. The paper's benchmark scores use larger models, long packed sequences, a different batch and learning-rate schedule, and task-specific evaluation. This runner uses 256-token truncation, a minimal ChatML template, a constant learning rate, and a small viewer-derived dataset split. Its NLL is a diagnostic for the local comparison.

## Initial GPU check

The three runs used the pinned base checkpoint on an RTX 5070 Ti with PyTorch 2.11.0+cu130, Transformers 5.9.0, seed 42, and 12 optimizer steps. After truncation, 23 train rows and 8 held-out rows were usable. Each run started at the same held-out NLL of 1.2974 over 736 supervised tokens. Lower NLL is better.

| Method | Clean / matching steps | Final held-out NLL | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Vanilla SFT | 12 / 0 | 1.8876 | 10.19 GiB |
| Frozen-base distillation, T=1 | 10 / 2 | 1.8989 | 11.30 GiB |
| Ambient SFT, T=2.5 | 10 / 2 | 1.9393 | 11.30 GiB |

All three runs increased NLL on this tiny held-out split. The check exercised each training path and measured memory use. Input file SHA-256 values for this check are `6a99c185deeec274b0c09b60ac2c8f981ecd1883cfaaf5ce898b32c374feeea3` (train) and `2621b06e2804211a87482504f034efc9ae0dbd231867a4b6c2d6a65d41a70ca2` (held-out).

The MIT license covers this repository's implementation. The model and dataset come from their respective publishers and are fetched separately.

## Follow-up experiments

- [IFEval output-budget check](docs/experiments/ifeval-output-budget.md) compares 512 and 2,048 output tokens on the original checkpoints.
- [Local benchmark protocol](docs/experiments/local-benchmark-protocol.md) gives the pinned data, training commands, and evaluation setup.
- [Experiment progress](PLAN.md) records the larger same-condition comparison as seed results are validated.

For larger samples, pass `--parquet path/to/train-shard.parquet` to `scripts/prepare_dolci.py` to read a local Dolci shard with PyArrow. The flag can be repeated to read multiple shards in order.
