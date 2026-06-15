from __future__ import annotations

import unittest
from pathlib import Path

from pointnet_final.data import Sample, stratified_train_val_split


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            Sample(f"class_{label}_{index:04d}", Path(f"{label}/{index}.txt"), label)
            for label in range(3)
            for index in range(20)
        ]

    def test_split_is_reproducible_disjoint_and_complete(self) -> None:
        train_a, val_a = stratified_train_val_split(self.samples, val_fraction=0.2, seed=2026)
        train_b, val_b = stratified_train_val_split(self.samples, val_fraction=0.2, seed=2026)

        self.assertEqual([sample.sample_id for sample in train_a], [sample.sample_id for sample in train_b])
        self.assertEqual([sample.sample_id for sample in val_a], [sample.sample_id for sample in val_b])

        train_ids = {sample.sample_id for sample in train_a}
        val_ids = {sample.sample_id for sample in val_a}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, {sample.sample_id for sample in self.samples})

    def test_each_class_is_represented_in_both_splits(self) -> None:
        train, val = stratified_train_val_split(self.samples, val_fraction=0.1, seed=2026)
        for label in range(3):
            self.assertTrue(any(sample.label == label for sample in train))
            self.assertTrue(any(sample.label == label for sample in val))


if __name__ == "__main__":
    unittest.main()
