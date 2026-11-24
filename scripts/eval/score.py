"""Score all responses and compute paired differences for the local experiment."""

import argparse
import json
import logging
import math
import random
import re
import sys
import types
from pathlib import Path

from generate import DATA, OLMES, read_jsonl


class VerifierErrors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(record.getMessage())


def load_instruction_verifier():
    # OLMES imports lm_eval.utils only to get this logger. Use the original scorer
    # with a local logger so unrelated harness extras are unnecessary.
    logger = logging.getLogger("local.ifeval")
    logger.setLevel(logging.ERROR)
    errors = VerifierErrors()
    logger.addHandler(errors)
    logger.propagate = False
    dependency = types.ModuleType("lm_eval")
    utilities = types.ModuleType("lm_eval.utils")
    utilities.eval_logger = logger
    dependency.utils = utilities
    sys.modules[dependency.__name__] = dependency
    sys.modules[utilities.__name__] = utilities
    sys.path.insert(0, str(OLMES))
    from oe_eval.dependencies.ifeval.utils import process_results
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    return process_results, errors


def gsm_answer(text):
    # GSM8K._extract_answer and _clean_short_answer in the pinned OLMES task.
    clean = re.sub(r"(\d),(\d)", r"\1\2", text)
    values = re.findall(r"[-+]?\d*\.\d+|\d+", clean)
    return values[-1] if values else clean.strip().lower()


def score(condition_dir, examples, verifier):
    summary = {}
    checks = {}
    for task, rows in examples.items():
        outputs = read_jsonl(condition_dir / f"{task}.jsonl")
        expected_ids = [row["key"] for row in rows] if task == "ifeval" else list(range(len(rows)))
        if [output["id"] for output in outputs] != expected_ids:
            raise ValueError(f"Response IDs differ from {task} dataset for {condition_dir.name}")
        decisions = []
        for row, output in zip(rows, outputs, strict=True):
            if task == "ifeval":
                metrics = verifier(row, [output["text"]])
            else:
                correct = gsm_answer(output["text"]) == gsm_answer(row["answer"].rsplit("####", 1)[-1])
                metrics = {"exact_match": correct}
            decisions.append({"id": output["id"], **metrics})
        checks[task] = decisions
        primary = "prompt_level_loose_acc" if task == "ifeval" else "exact_match"
        task_scores = {
            "n": len(rows), "correct": sum(row[primary] for row in decisions),
            "max_tokens_reached": sum(row["finish"] == "length" for row in outputs),
            "generated_tokens": sum(row["tokens"] for row in outputs),
        }
        task_scores[primary] = task_scores["correct"] / len(rows)
        if task == "ifeval":
            task_scores["prompt_level_strict_acc"] = sum(row["prompt_level_strict_acc"] for row in decisions) / len(rows)
        summary[task] = task_scores
    return summary, checks


def paired(first, second, metric):
    wins = losses = 0
    diffs = []
    for a, b in zip(first, second, strict=True):
        assert a["id"] == b["id"]
        diff = int(b[metric]) - int(a[metric])
        wins += diff == 1
        losses += diff == -1
        diffs.append(diff)
    n = len(diffs)
    discordant = wins + losses
    p_value = min(1, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant)
    delta = (wins - losses) / n
    rng = random.Random(123)
    replicates = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(5000))
    return {
        "delta_pp": 100 * delta, "wins": wins, "losses": losses, "mcnemar_p": p_value,
        "paired_bootstrap_95_pp": [100 * replicates[125], 100 * replicates[4874]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task", choices=("ifeval", "gsm8k", "both"), default="both")
    args = parser.parse_args()
    tasks = ("ifeval", "gsm8k") if args.task == "both" else (args.task,)
    examples = {task: read_jsonl(DATA / f"{task}.jsonl") for task in tasks}
    verifier, errors = load_instruction_verifier() if "ifeval" in tasks else (None, None)
    random.seed(0)
    all_scores, all_checks = {}, {}
    for condition in ("sft", "kd", "ambient"):
        all_scores[condition], all_checks[condition] = score(args.output_root / condition, examples, verifier)
        print(condition, json.dumps(all_scores[condition]), flush=True)
    if errors is not None and errors.errors:
        raise RuntimeError(f"{len(errors.errors)} IFEval verifier errors; first: {errors.errors[0]}")
    comparisons = {}
    benchmark_metrics = {"ifeval": "prompt_level_loose_acc", "gsm8k": "exact_match"}
    for first, second in (("sft", "kd"), ("sft", "ambient"), ("kd", "ambient")):
        comparisons[f"{second}_vs_{first}"] = {
            task: paired(all_checks[first][task], all_checks[second][task], benchmark_metrics[task])
            for task in tasks
        }
    for condition, checks in all_checks.items():
        path = args.output_root / condition / "scored.json"
        path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    result = {"scores": all_scores, "paired": comparisons, "ifeval_verifier_errors": 0}
    (args.output_root / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
