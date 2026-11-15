from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from actionrankwm.io import iter_jsonl

ROOT = Path(__file__).resolve().parents[1]


class GzipJsonlTests(unittest.TestCase):
    def test_iter_jsonl_reads_plain_and_gzip(self) -> None:
        rows = [{"a": 1}, {"b": [2, 3]}]
        payload = "\n".join(json.dumps(r) for r in rows) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            plain = Path(temporary) / "x.jsonl"
            plain.write_text(payload, encoding="utf-8")
            packed = Path(temporary) / "x.jsonl.gz"
            packed.write_bytes(gzip.compress(payload.encode("utf-8")))
            self.assertEqual(list(iter_jsonl(plain)), rows)
            self.assertEqual(list(iter_jsonl(packed)), rows)


class ReproducePaperTablesTests(unittest.TestCase):
    def test_released_records_reproduce_headline_results(self) -> None:

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "reproduce_paper_tables.py"), "--fast", "--json", str(report)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            results = json.loads(report.read_text(encoding="utf-8"))["results"]
        for macro, value in {
            "EBPooledHFUnsafe": "92",
            "BridgeHFSelectedTotal": "37",
            "BridgeHFFailTotal": "17",
            "BridgeFailureTotal": "39",
            "EBFourBPair": "0.836",
            "EBNineBPair": "0.753",
            "UFBinFourPair": "0.894",
            "VENineMAE": "0.284",
        }.items():
            self.assertEqual(results[macro], value)


if __name__ == "__main__":
    unittest.main()
