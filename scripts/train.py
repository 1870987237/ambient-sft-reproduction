"""Run a single-device SFT or Ambient SFT experiment on local JSONL data."""

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
IGNORE_INDEX = -100
CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}"
    "{{ '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def encode_example(tokenizer, messages, max_length):
    encoded = tokenizer.apply_chat_template(
        messages, chat_template=CHATML, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True,
    )
    tokens = encoded["input_ids"][:max_length]
    mask = encoded["assistant_masks"][:max_length]
    if not any(mask[1:]):
        return None
    labels = [token if supervised else IGNORE_INDEX for token, supervised in zip(tokens, mask)]
    return torch.tensor([tokens]), torch.tensor([labels])


def load_examples(path, tokenizer, max_length):
    examples = []
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if not line.strip():
                continue
            count += 1
            try:
                example = encode_example(tokenizer, json.loads(line)["messages"], max_length)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid example {count} in {path.name}") from error
            if example is not None:
                examples.append(example)
    if not examples:
        raise ValueError(f"{path.name} has no supervised tokens after truncation")
    return examples, {"rows": count, "usable": len(examples), "sha256": digest.hexdigest()}


def causal_targets(labels):
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    return mask, targets[mask]


def supervised_logits(logits, mask):
    return logits[:, :-1, :][mask].float()


def matching_loss(student_logits, teacher_logits, temperature):
    teacher_probs = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


@torch.no_grad()
def reference_logits(teacher, tokens, mask, offload=False):
    if offload:
        # Keep the original frozen tensors on the host and discard the GPU views
        # after the forward pass. The teacher needs no copy back to the host.
        parameters = {name: value.to(tokens.device, non_blocking=True)
                      for name, value in teacher.named_parameters()}
        buffers = {name: value.to(tokens.device, non_blocking=True)
                   for name, value in teacher.named_buffers()}
        outputs = torch.func.functional_call(teacher, (parameters, buffers), (tokens,), {"use_cache": False})
    else:
        outputs = teacher(tokens, use_cache=False)
    return supervised_logits(outputs.logits, mask)


def use_matching(clean_probability, branch_rng):
    return clean_probability < 1 and branch_rng.random() >= clean_probability


def shuffled_examples(examples, rng):
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for index in order:
            yield examples[index]


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(model, examples, device):
    model.eval()
    summed_loss = 0.0
    token_count = 0
    for tokens, labels in examples:
        tokens, labels = tokens.to(device), labels.to(device)
        mask, targets = causal_targets(labels)
        with autocast(device):
            logits = model(tokens, use_cache=False).logits
        summed_loss += F.cross_entropy(supervised_logits(logits, mask), targets, reduction="sum").item()
        token_count += targets.numel()
    return {"nll": summed_loss / token_count, "tokens": token_count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sft", "ambient"), required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--clean-probability", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--low-vram", action="store_true", help="BF16 full weights, 8-bit AdamW, and a CPU teacher")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.max_length < 2:
        parser.error("steps must be positive and max-length must be at least 2")
    if not 0 < args.clean_probability <= 1:
        parser.error("clean-probability must be in (0, 1]")
    if any(not math.isfinite(x) or x <= 0 for x in (args.temperature, args.learning_rate)):
        parser.error("temperature and learning-rate must be finite and positive")
    if args.train_data.resolve() == args.eval_data.resolve():
        parser.error("train and evaluation files must be different")
    if args.save_dir is not None and args.save_dir.exists():
        parser.error("save-dir must be a new directory")
    if args.low_vram and args.device != "cuda":
        parser.error("low-vram requires CUDA")
    return args


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("The CUDA device and PyTorch build must support bfloat16")
    torch.manual_seed(args.seed)
    branch_rng = random.Random(args.seed + 1)
    data_rng = random.Random(args.seed)
    revision = args.revision or (BASE_REVISION if args.model == BASE_MODEL else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    chat_tokens = ("<|im_start|>", "<|im_end|>")
    if any(token not in tokenizer.get_added_vocab() for token in chat_tokens):
        raise ValueError("This runner requires a tokenizer with ChatML control tokens")
    train, train_info = load_examples(args.train_data, tokenizer, args.max_length)
    validation, validation_info = load_examples(args.eval_data, tokenizer, args.max_length)
    if train_info["sha256"] == validation_info["sha256"]:
        raise ValueError("Train and evaluation files contain identical bytes")

    probability = args.clean_probability if args.mode == "ambient" else 1.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    student_dtype = torch.bfloat16 if args.low_vram else torch.float32
    student = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=student_dtype, attn_implementation="sdpa",
    ).to(device)
    teacher = None
    teacher_device = torch.device("cpu") if args.low_vram else device
    if probability < 1:
        teacher_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, dtype=teacher_dtype, attn_implementation="sdpa",
        ).to(teacher_device).eval().requires_grad_(False)
    optimizer_kwargs = dict(lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0)
    if args.low_vram:
        from bitsandbytes.optim import AdamW8bit

        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = AdamW8bit(student.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), foreach=False, **optimizer_kwargs)
    warmup_steps = int(0.03 * args.steps) if args.schedule == "linear" else 0
    scheduler = None
    if args.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.steps)
    before = evaluate(student, validation, device)
    student.train()
    counts = {"clean": 0, "matching": 0}
    examples = shuffled_examples(train, data_rng)
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        tokens, labels = (tensor.to(device) for tensor in next(examples))
        mask, targets = causal_targets(labels)
        optimizer.zero_grad(set_to_none=True)
        matching = teacher is not None and use_matching(probability, branch_rng)
        reference = reference_logits(teacher, tokens, mask, args.low_vram) if matching else None
        with autocast(device):
            logits = supervised_logits(student(tokens, use_cache=False).logits, mask)
        if matching:
            branch = "matching"
            loss = matching_loss(logits, reference, args.temperature)
        else:
            branch = "clean"
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {branch} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        counts[branch] += 1
        print(json.dumps({
            "step": step, "branch": branch, "loss": loss.item(), "tokens": targets.numel(),
            "learning_rate": learning_rate,
        }), flush=True)
        del logits, loss, reference

    elapsed = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    if teacher is not None and any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("The frozen teacher received gradients")
    after = evaluate(student, validation, device)
    summary = {
        "mode": args.mode, "model": args.model, "revision": revision, "seed": args.seed,
        "device": str(device), "torch": torch.__version__, "max_length": args.max_length,
        "optimizer": type(optimizer).__name__, "student_dtype": str(student_dtype),
        "gradient_checkpointing": args.low_vram,
        "teacher_device": str(teacher_device) if teacher is not None else None,
        "steps": args.steps, "learning_rate": args.learning_rate,
        "schedule": args.schedule, "warmup_steps": warmup_steps,
        "temperature": args.temperature, "clean_probability": probability,
        "branches": counts, "train": train_info, "validation": validation_info,
        "eval_before": before, "eval_after": after, "train_seconds": elapsed,
    }
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=False)
        student.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        tokenizer.chat_template = CHATML
        student.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
