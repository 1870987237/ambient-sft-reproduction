"""Generate on the fixed IFEval and GSM8K items for the local experiment."""

import argparse
import json
import os
import runpy
import time
from pathlib import Path


WORK = Path(__file__).resolve().parents[2] / "work"
DATA = WORK / "benchmark-data"
OLMES = WORK / "olmes-upstream"
GSM_STOPS = ["Question:", "</s>", "<|im_end|>", "\n\n"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_prompts(tokenizer, task):
    rows = read_jsonl(DATA / f"{task}.jsonl")
    if task == "ifeval":
        template = (DATA / "olmo3-chat.jinja").read_text(encoding="utf-8")
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
        ) for row in rows]
        return [row["key"] for row in rows], prompts, 2048, []

    demo_file = OLMES / "oe_eval" / "tasks" / "fewshot_sources.py"
    demos = runpy.run_path(str(demo_file))["FEWSHOT_SOURCES"]["STD:GSM8k"]
    assert len(demos) == 8
    prefix = "\n\n".join(f"Question: {row['question']}\nAnswer: {row['answer']}" for row in demos)
    prompts = [f"{prefix}\n\nQuestion: {row['question']}\nAnswer:" for row in rows]
    return list(range(len(rows))), prompts, 512, GSM_STOPS


def vllm_generation(model, tokenizer, prompts, batch_size, token_limit, stop_strings, stop_ids):
    from vllm import SamplingParams

    settings = SamplingParams(
        temperature=0.0, max_tokens=token_limit, stop=stop_strings,
        stop_token_ids=stop_ids, skip_special_tokens=True,
    )
    for offset in range(0, len(prompts), batch_size):
        chunk = prompts[offset : offset + batch_size]
        token_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        inputs = [{"prompt_token_ids": ids} for ids in token_ids]
        outputs = model.generate(inputs, settings, use_tqdm=False)
        for output in outputs:
            candidate = output.outputs[0]
            yield candidate.text, len(candidate.token_ids), candidate.finish_reason
        print(f"generated {min(offset + batch_size, len(prompts))}/{len(prompts)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-attention", default="FLASH_ATTN")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", choices=("ifeval", "gsm8k", "both"), default="both")
    parser.add_argument("--ifeval-max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.ifeval_max_tokens <= 2048:
        parser.error("IFEval output budget must be between 1 and 2048")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and optional example limit must be positive")

    from transformers import AutoTokenizer

    checkpoint = str(args.model.resolve())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    stop_ids = sorted({tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")})
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_CACHE_ROOT", str(WORK / "vllm-cache"))
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM

    model = LLM(
        model=checkpoint, dtype="bfloat16", max_model_len=3072,
        max_num_seqs=128, max_num_batched_tokens=8192,
        enforce_eager=True, gpu_memory_utilization=0.72, enable_prefix_caching=True,
        generation_config="vllm",
        attention_config={"backend": args.vllm_attention},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ("ifeval", "gsm8k") if args.task == "both" else (args.task,)
    for task in tasks:
        ids, prompts, limit, stops = build_prompts(tokenizer, task)
        if task == "ifeval":
            limit = args.ifeval_max_tokens
        ids, prompts = ids[:args.limit], prompts[:args.limit]
        started = time.perf_counter()
        generated = vllm_generation(model, tokenizer, prompts, args.batch_size, limit, stops, stop_ids)
        filename = args.output_dir / f"{task}.jsonl"
        with filename.open("w", encoding="utf-8") as output:
            for item_id, (text, tokens, finish) in zip(ids, generated, strict=True):
                row = {"id": item_id, "text": text, "tokens": tokens, "finish": finish}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{task} {len(ids)} prompts in {time.perf_counter() - started:.1f}s via vLLM", flush=True)


if __name__ == "__main__":
    main()
