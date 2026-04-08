from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas_training.util import json_ready, write_json


class TrainingUtilTest(unittest.TestCase):
    def test_json_ready_converts_paths_recursively(self) -> None:
        payload = {
            "path": Path("results/example"),
            "nested": {"items": [Path("foo"), ("bar", Path("baz"))]},
        }
        ready = json_ready(payload)
        self.assertEqual(ready["path"], "results/example")
        self.assertEqual(ready["nested"]["items"][0], "foo")
        self.assertEqual(ready["nested"]["items"][1], ["bar", "baz"])

    def test_write_json_writes_normalized_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "payload.json"
            write_json(output, {"path": Path("results/example")})
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["path"], "results/example")


if __name__ == "__main__":
    unittest.main()
