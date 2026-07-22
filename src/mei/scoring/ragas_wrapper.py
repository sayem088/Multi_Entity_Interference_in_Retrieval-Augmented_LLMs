"""
RAGAS Faithfulness Wrapper Implementation.
Evaluates propositional faithfulness (checking if stated facts exist somewhere in context).
"""

from typing import Dict, Any

def compute_ragas_faithfulness(response_text: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes propositional faithfulness score.
    Verifies if individual facts stated in response appear anywhere in context.
    """
    entity_map = ground_truth.get("entity_map", {})
    
    # Flatten all retrieved values into context values pool
    context_values = set()
    for attrs in entity_map.values():
        for k, v in attrs.items():
            context_values.add(str(v).strip().lower())

    # Count how many stated values exist somewhere in context
    extracted_facts = 0
    grounded_facts = 0

    lines = response_text.split("\n")
    for line in lines:
        words = line.split()
        for word in words:
            clean_word = word.strip(",.!:()").lower()
            if len(clean_word) > 2 and clean_word not in ["has", "is", "the", "and", "for", "with", "site", "entity"]:
                extracted_facts += 1
                if clean_word in context_values or any(cv in clean_word for cv in context_values):
                    grounded_facts += 1

    faithfulness = (grounded_facts / extracted_facts) if extracted_facts > 0 else 1.0

    return {
        "ragas_faithfulness": min(1.0, faithfulness)
    }
