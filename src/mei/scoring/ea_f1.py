"""
Entity-Attributed F1 (EA-F1) Scorer Implementation.
Follows Lin et al. (2025) MEBench definition.
"""

from typing import Dict, Any
from mei.scoring.ber_k import extract_triples_from_text

def compute_ea_f1(response_text: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 score for entity-attributed claims.
    """
    entity_map = ground_truth.get("entity_map", {})
    
    # Total ground truth entity-attribute triples
    gt_triples = set()
    for ent_name, attrs in entity_map.items():
        for k, v in attrs.items():
            if k != "name":
                gt_triples.add((ent_name, k, str(v).strip()))

    extracted_claims = extract_triples_from_text(response_text, ground_truth)
    pred_triples = set((ent, k, str(v).strip()) for ent, k, v in extracted_claims)

    correct_predictions = pred_triples.intersection(gt_triples)

    precision = len(correct_predictions) / len(pred_triples) if pred_triples else 0.0
    recall = len(correct_predictions) / len(gt_triples) if gt_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "ea_precision": precision,
        "ea_recall": recall,
        "ea_f1": f1
    }
