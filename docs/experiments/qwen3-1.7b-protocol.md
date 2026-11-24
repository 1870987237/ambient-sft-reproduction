# Qwen3-1.7B single-GPU comparison

These runs use the paper's [Qwen3-1.7B-Base model](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1). The base checkpoint is evaluated in the same harness as the three trained checkpoints. The dataset split, sequence cap, seed and schedule match the existing 0.6B seed-42 experiment. [Frozen split and evaluation protocol](local-benchmark-protocol.md)

The RTX 5070 Ti run trains all student parameters with BF16 weights and AdamW with 8-bit optimizer state. The frozen teacher resides in system RAM. The runner sends a transient BF16 copy of its original tensors to CUDA on matching steps. The student reuses the memory after the teacher forward. All three conditions use `--low-vram`, even when the selected method has no teacher.

```text
python -m pip install bitsandbytes
python scripts/train.py --mode sft --low-vram --model work/models/qwen3-1.7b-base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --train-data work/dolci-expanded/train.jsonl --eval-data work/dolci-expanded/eval.jsonl --steps 2048 --schedule linear --max-length 384 --seed 42 --save-dir work/17b/sft
```

For the two matching runs, change `--mode` to `ambient`, set `--temperature` to `1` or `2.5`, and use separate save directories. For the base condition, evaluate the original model directory. When scoring all four directories, use `python scripts/eval/score.py work/17b-benchmarks --conditions base sft kd ambient`.

This comparison fixes 4,000 Dolci candidates, 2,048 updates, one example per update, and 384 input tokens. The paper's 1% Qwen3-1.7B experiment used about 21,500 examples for ten epochs, a 32,768-token sequence length, and a global batch of 1,048,576 tokens. [Paper Appendix C.2](https://arxiv.org/html/2611.23229v1#A3.SS2)
