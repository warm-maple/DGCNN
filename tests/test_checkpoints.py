from __future__ import annotations

import unittest

from pointnet_final.checkpoints import rank_top_checkpoints


class CheckpointRankingTests(unittest.TestCase):
    def test_keeps_highest_five_balanced_scores(self) -> None:
        records = [
            {"epoch": epoch, "instance_acc": score, "class_acc": score}
            for epoch, score in enumerate([0.70, 0.80, 0.75, 0.90, 0.85, 0.95], start=1)
        ]

        ranked = rank_top_checkpoints(records, limit=5)

        self.assertEqual([record["epoch"] for record in ranked], [6, 4, 5, 2, 3])


if __name__ == "__main__":
    unittest.main()
