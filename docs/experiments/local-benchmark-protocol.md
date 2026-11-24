# Local IFEval and GSM8K protocol

The runner evaluates local Qwen3 checkpoints on all 541 IFEval prompts and 1,319 GSM8K test problems. It uses the OLMo 3 chat template for zero-shot IFEval and the eight fixed OLMES demonstrations for plain-text GSM8K. The primary metrics are IFEval prompt-level loose accuracy and GSM8K exact match. Greedy decoding uses 2,048 output tokens for IFEval and 512 for GSM8K. The output-budget check changes the IFEval cap to 512.

## Benchmark inputs and runtime

Use a working CUDA vLLM installation for generation. The local experiments use vLLM 0.21.0, BF16, and FlashAttention on an RTX 5070 Ti. Prepare the pinned inputs and scoring code from the repository root.

```text
python scripts/eval/prepare_assets.py
git clone --filter=blob:none https://github.com/allenai/olmes.git work/olmes-upstream
git -C work/olmes-upstream switch --detach 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e
```

Install the auxiliary IFEval dependencies in the Python environment used for scoring. A separate scoring environment also works.

```text
python -m pip install nltk==3.8.1 "spacy>=3.8,<4" langdetect immutabledict emoji syllapy
python -m nltk.downloader punkt stopwords
```

```text
python scripts/eval/generate.py --model work/checkpoints/seed-42/sft --output-dir work/scores/seed-42/sft --batch-size 2048
python scripts/eval/generate.py --model work/checkpoints/seed-42/kd --output-dir work/scores/seed-42/kd --batch-size 2048
python scripts/eval/generate.py --model work/checkpoints/seed-42/ambient --output-dir work/scores/seed-42/ambient --batch-size 2048
python scripts/eval/score.py work/scores/seed-42
```

For the 512-token IFEval condition, use `--task ifeval --ifeval-max-tokens 512` for generation and `--task ifeval` for scoring, with a separate output directory. The scorer checks response IDs before producing aggregate and paired results.

## Training split for the 2,048-step comparison

The split came from `data/train-00000-of-00015.parquet` in [this Dolci dataset revision](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT/tree/bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221). Its SHA-256 is `8a671c8118a611338059f931e64defe4817202130c00dc547f109d07c86b84a6`. The first 4,000 eligible rows are training candidates and the next 256 are held out. There were zero exact normalized matches with either benchmark among the candidate user turns checked. Use PyArrow to extract the split.

```text
python -m pip install pyarrow
python scripts/prepare_dolci.py --parquet path/to/train-00000-of-00015.parquet --train-count 4000 --eval-count 256 --output-dir work/dolci-expanded
```

The expected SHA-256 checksums are `1cffb7f45f1b41042de9bac50b518a6895abbe672559c96fefd87ef988ef472d` for training and `2cefeb59d5bf522fef43d53df8d2ce94cf29b7011e3ee9043323a297667acfd6` for the holdout.

Each run uses Qwen3-0.6B-Base, 2,048 steps, a maximum training length of 384 tokens, one example per update, and the same linear schedule with a 3% warmup. The three seeds are 42, 43, and 44. Use the same seed and run settings for the three methods within each comparison.

```text
python scripts/train.py --mode sft --train-data work/dolci-expanded/train.jsonl --eval-data work/dolci-expanded/eval.jsonl --steps 2048 --schedule linear --max-length 384 --seed 42 --save-dir work/checkpoints/seed-42/sft
python scripts/train.py --mode ambient --temperature 1 --train-data work/dolci-expanded/train.jsonl --eval-data work/dolci-expanded/eval.jsonl --steps 2048 --schedule linear --max-length 384 --seed 42 --save-dir work/checkpoints/seed-42/kd
python scripts/train.py --mode ambient --temperature 2.5 --train-data work/dolci-expanded/train.jsonl --eval-data work/dolci-expanded/eval.jsonl --steps 2048 --schedule linear --max-length 384 --seed 42 --save-dir work/checkpoints/seed-42/ambient
```
