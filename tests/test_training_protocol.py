from __future__ import annotations

import unittest
from pathlib import Path

from pointnet_final.train import DEFAULT_SELECTION_SPLIT


class TrainingProtocolTests(unittest.TestCase):
    def test_checkpoint_selection_uses_validation_split(self) -> None:
        self.assertEqual(DEFAULT_SELECTION_SPLIT, "val_split")

    def test_training_module_does_not_reference_official_test_split(self) -> None:
        source = Path("pointnet_final/train.py").read_text(encoding="utf-8")
        self.assertNotIn("list_modelnet_split", source)
        self.assertNotIn('"official_test"', source)


if __name__ == "__main__":
    unittest.main()
