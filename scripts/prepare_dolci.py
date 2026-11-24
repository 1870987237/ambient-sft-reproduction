"""Create a local Dolci split from the public viewer or PyArrow-readable shards."""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "allenai/Dolci-Instruct-SFT"
VIEWER = "https://datasets-server.huggingface.co/rows"
ROLES = {"system", "user", "assistant"}


def viewer_rows(offset):
    while True:
        query = urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": 100,
        })
        with urlopen(f"{VIEWER}?{query}", timeout=45) as response:
            rows = json.load(response)["rows"]
        if not rows:
            return
        yield from (item["row"] for item in rows)
        offset += len(rows)


def parquet_rows(paths):
    import pyarrow.parquet as pq

    for path in paths:
        with pq.ParquetFile(path) as shard:
            for batch in shard.iter_batches(batch_size=1024, columns=["id", "messages"]):
                yield from batch.to_pylist()


def fetch_split(train_count, eval_count, offset, parquet_paths=None):
    seen_ids = set()
    seen_messages = set()
    examples = []
    limit = train_count + eval_count
    source = islice(parquet_rows(parquet_paths), offset, None) if parquet_paths else viewer_rows(offset)

    for row in source:
        messages = row.get("messages")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not messages or row_id in seen_ids:
            continue
        if any(
            m.get("role") not in ROLES or not isinstance(m.get("content"), str)
            or m.get("function_calls") or m.get("functions")
            for m in messages
        ):
            continue
        pairs = tuple((m["role"], m["content"]) for m in messages)
        if pairs in seen_messages or not any(r == "assistant" and c.strip() for r, c in pairs):
            continue
        seen_ids.add(row_id)
        seen_messages.add(pairs)
        examples.append({"id": row_id, "messages": [
            {"role": role, "content": content} for role, content in pairs
        ]})
        if len(examples) == limit:
            break

    if len(examples) < limit:
        raise RuntimeError(f"The data source ended after {len(examples)} usable examples (requested {limit})")
    return examples[:train_count], examples[train_count:]


def write_jsonl(path, examples):
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples).encode("utf-8")
    path.write_bytes(data)
    print(f"{path.name}: {len(examples)} examples, sha256={hashlib.sha256(data).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dolci"))
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--parquet", type=Path, action="append", help="Local shard, repeat in source order")
    args = parser.parse_args()
    if args.train_count < 1 or args.eval_count < 1 or args.offset < 0:
        parser.error("counts must be positive and offset must be nonnegative")
    train, evaluation = fetch_split(args.train_count, args.eval_count, args.offset, args.parquet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "eval.jsonl", evaluation)


if __name__ == "__main__":
    main()
