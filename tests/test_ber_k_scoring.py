"""
Unit tests for BER_k scoring logic: correct bindings, binding errors (swaps), and hallucinations.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.scoring.ber_k import compute_ber_k

class TestBERkScoring(unittest.TestCase):
    def setUp(self):
        self.sample_ground_truth = {
            "scenario_id": "test_001",
            "entity_map": {
                "St Mary Hospital": {
                    "capacity": "250 beds",
                    "opening_hours": "24/7"
                },
                "City General Hospital": {
                    "capacity": "500 beds",
                    "opening_hours": "08:00-20:00"
                }
            }
        }

    def test_clean_correct_response(self):
        text = (
            "St Mary Hospital has capacity as 250 beds, opening_hours as 24/7.\n"
            "City General Hospital has capacity as 500 beds, opening_hours as 08:00-20:00."
        )
        result = compute_ber_k(text, self.sample_ground_truth)
        self.assertEqual(result["ber_k"], 0.0)
        self.assertGreaterEqual(result["correct_bindings"], 2)
        self.assertEqual(result["binding_errors"], 0)

    def test_binding_error_swap_response(self):
        # St Mary Hospital incorrectly stated as having 500 beds (value belongs to City General!)
        text = (
            "St Mary Hospital has capacity as 500 beds, opening_hours as 24/7.\n"
            "City General Hospital has capacity as 250 beds, opening_hours as 08:00-20:00."
        )
        result = compute_ber_k(text, self.sample_ground_truth)
        self.assertGreater(result["binding_errors"], 0)
        self.assertGreater(result["ber_k"], 0.0)

    def test_pure_hallucination_exclusion(self):
        # "999 beds" is absent from context entirely (hallucination, excluded from BER_k denominator)
        text = (
            "St Mary Hospital has capacity as 999 beds."
        )
        result = compute_ber_k(text, self.sample_ground_truth)
        self.assertEqual(result["hallucinations"], 1)
        self.assertEqual(result["binding_errors"], 0)
        self.assertEqual(result["ber_k"], 0.0)

if __name__ == "__main__":
    unittest.main()
