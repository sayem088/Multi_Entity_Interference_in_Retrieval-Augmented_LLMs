"""
Unit tests for schema similarity (Jaccard calculation and mean pairwise calculation).
"""

import sys
import unittest
from pathlib import Path

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.schema.similarity import jaccard_similarity, pairwise_mean_similarity

class TestSchemaSimilarity(unittest.TestCase):
    def test_jaccard_exact_match(self):
        keys1 = ["capacity", "opening_hours", "phone"]
        keys2 = ["capacity", "opening_hours", "phone"]
        self.assertEqual(jaccard_similarity(keys1, keys2), 1.0)

    def test_jaccard_disjoint(self):
        keys1 = ["capacity", "emergency"]
        keys2 = ["cuisine", "takeaway"]
        self.assertEqual(jaccard_similarity(keys1, keys2), 0.0)

    def test_jaccard_partial_overlap(self):
        keys1 = ["name", "phone", "website", "capacity"]
        keys2 = ["name", "phone", "cuisine", "rating"]
        # Intersection: ["name", "phone"] -> 2
        # Union: ["name", "phone", "website", "capacity", "cuisine", "rating"] -> 6
        # 2/6 = 0.33333...
        self.assertAlmostEqual(jaccard_similarity(keys1, keys2), 2.0 / 6.0, places=3)

    def test_pairwise_mean_similarity(self):
        schemas = [
            ["name", "phone", "capacity"],
            ["name", "phone", "capacity"],
            ["name", "cuisine"]
        ]
        # Pair 1 (0, 1): J = 3/3 = 1.0
        # Pair 2 (0, 2): J = 1/4 = 0.25
        # Pair 3 (1, 2): J = 1/4 = 0.25
        # Mean = (1.0 + 0.25 + 0.25) / 3 = 1.5 / 3 = 0.50
        score = pairwise_mean_similarity(schemas)
        self.assertAlmostEqual(score, 0.50, places=3)

if __name__ == "__main__":
    unittest.main()
