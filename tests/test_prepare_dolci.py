import unittest
from unittest.mock import patch

from scripts.prepare_dolci import fetch_split


def entry(row_id, question, answer):
    return {"id": row_id, "messages": [
        {"role": "user", "content": question}, {"role": "assistant", "content": answer},
    ]}


class PrepareDolciTests(unittest.TestCase):
    def test_parquet_offset_and_duplicates(self):
        source = [
            entry("skip", "Skip", "Skipped"),
            entry("train", "Train", "Training answer"),
            entry("duplicate", "Train", "Training answer"),
            entry("eval", "Evaluate", "Evaluation answer"),
        ]
        with patch("scripts.prepare_dolci.parquet_rows", return_value=iter(source)):
            train, evaluation = fetch_split(1, 1, 1, ["local.parquet"])
        self.assertEqual([row["id"] for row in train], ["train"])
        self.assertEqual([row["id"] for row in evaluation], ["eval"])

    def test_short_viewer_snapshot_fails(self):
        source = [entry("train", "Train", "Training answer"), entry("empty", "Question", " ")]
        with patch("scripts.prepare_dolci.viewer_rows", return_value=iter(source)):
            with self.assertRaisesRegex(RuntimeError, "1 usable examples"):
                fetch_split(1, 1, 0)
