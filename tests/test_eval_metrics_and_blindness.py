"""
Unit tests verifying Entity-Attributed F1 (EA-F1) and the Evaluation Blindness Paradox.
Demonstrates deterministically how un-scoped claim faithfulness (RAGAS) remains invariant
to cross-entity attribute transpositions that are caught by EA-F1 and BER_k.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.scoring.ber_k import compute_ber_k
from mei.scoring.ea_f1 import compute_ea_f1
from mei.scoring.ragas_wrapper import compute_ragas_faithfulness


class TestEvalMetricsAndBlindness(unittest.TestCase):
    def setUp(self):
        self.ground_truth = {
            "scenario_id": "test_clinical_swap_001",
            "entity_map": {
                "Protocol Alpha": {
                    "intervention_dosage": "200mg",
                    "sample_size": "820",
                },
                "Protocol Beta": {
                    "intervention_dosage": "5.4mg",
                    "sample_size": "240",
                },
            },
        }

    def test_exact_binding_and_transposition_blindness(self):
        clean_response = (
            "Protocol Alpha has intervention_dosage: 200mg, sample_size: 820.\n"
            "Protocol Beta has intervention_dosage: 5.4mg, sample_size: 240."
        )
        swapped_response = (
            "Protocol Alpha has intervention_dosage: 5.4mg, sample_size: 240.\n"
            "Protocol Beta has intervention_dosage: 200mg, sample_size: 820."
        )

        # Clean response yields perfect deterministic entity binding
        ber_clean = compute_ber_k(clean_response, self.ground_truth)
        ea_clean = compute_ea_f1(clean_response, self.ground_truth)
        ragas_clean = compute_ragas_faithfulness(clean_response, self.ground_truth)

        self.assertEqual(ber_clean["ber_k"], 0.0)
        self.assertEqual(ea_clean["ea_f1"], 1.0)
        self.assertGreater(ragas_clean["ragas_faithfulness"], 0.0)

        # Swapped response collapses deterministic binding while leaving RAGAS invariant
        ber_swap = compute_ber_k(swapped_response, self.ground_truth)
        ea_swap = compute_ea_f1(swapped_response, self.ground_truth)
        ragas_swap = compute_ragas_faithfulness(swapped_response, self.ground_truth)

        self.assertEqual(ber_swap["ber_k"], 1.0)
        self.assertEqual(ea_swap["ea_f1"], 0.0)
        self.assertEqual(ea_swap["ea_recall"], 0.0)
        self.assertAlmostEqual(
            ragas_swap["ragas_faithfulness"], ragas_clean["ragas_faithfulness"], places=4
        )
        self.assertGreater(ragas_swap["ragas_faithfulness"] - ea_swap["ea_recall"], 0.25)


if __name__ == "__main__":
    unittest.main()
