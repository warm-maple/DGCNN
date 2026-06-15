from __future__ import annotations

import unittest

from pointnet_final.evaluate_official import OFFICIAL_TEST_CACHE_NAME


class OfficialEvaluationTests(unittest.TestCase):
    def test_official_test_has_a_distinct_cache_name(self) -> None:
        self.assertEqual(OFFICIAL_TEST_CACHE_NAME, "official_test")


if __name__ == "__main__":
    unittest.main()
