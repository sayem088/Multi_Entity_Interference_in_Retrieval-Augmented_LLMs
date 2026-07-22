"""
Binding Error Rate (BER_k) Metric Implementation.

Definition:
BER_k = (attribute statements in response that are factually present in context but bound to wrong entity)
        --------------------------------------------------------------------------------------------------
        (total attribute statements referencing retrieved facts: correct bindings + binding errors)

Note: Pure hallucinations (values absent from context entirely) are excluded from BER_k's denominator.
"""

import re
from typing import Dict, Any, List, Tuple, Set

def _entity_matches_text(ent_name: str, text: str) -> bool:
    """Checks if entity name or its unique core words appear in text."""
    ent_lower = ent_name.lower()
    text_lower = text.lower()
    
    if ent_lower in text_lower:
        return True
        
    # Extract distinct non-generic words (ignore common categories like hospital, university, company, etc.)
    stopwords = {"hospital", "university", "company", "bank", "restaurant", "site", "park", "pharmacy", "stadium", "the", "grand", "city"}
    words = [w for w in re.findall(r'\w+', ent_lower) if w not in stopwords]
    
    if words and all(w in text_lower for w in words):
        return True
        
    return False


def extract_triples_from_text(response_text: str, ground_truth: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Extracts (entity, attribute, value) claims from response_text matching the known ground_truth schema.
    """
    entity_map = ground_truth.get("entity_map", {})
    extracted_claims = []
    
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    # Collect all known attribute values across ground truth
    all_known_attrs = []
    for ent_name, attrs in entity_map.items():
        for k, v in attrs.items():
            if k != "name":
                all_known_attrs.append((ent_name, k, str(v).strip()))

    # Process line by line to determine entity context
    for line in lines:
        for ent_name in entity_map.keys():
            if _entity_matches_text(ent_name, line):
                # Search for any known attribute values in this line
                for orig_ent, k, val in all_known_attrs:
                    if val.lower() in line.lower():
                        claim = (ent_name, k, val)
                        if claim not in extracted_claims:
                            extracted_claims.append(claim)

    # Check for hallucinated attributes explicitly mentioned near an entity
    if not extracted_claims or "has" in response_text.lower():
        # Generic regex to capture "has <attribute> as <value>"
        matches = re.findall(r'(\w+[\w\s]*?)\s+has\s+(\w+)\s+as\s+([^.,\n]+)', response_text, re.IGNORECASE)
        for matched_ent, matched_attr, matched_val in matches:
            matched_ent_clean = matched_ent.strip()
            matched_val_clean = matched_val.strip()
            for ent_name in entity_map.keys():
                if _entity_matches_text(ent_name, matched_ent_clean):
                    claim = (ent_name, matched_attr.strip(), matched_val_clean)
                    if claim not in extracted_claims:
                        extracted_claims.append(claim)

    return extracted_claims


def compute_ber_k(response_text: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes BER_k score and detailed claim breakdown.
    """
    entity_map = ground_truth.get("entity_map", {})
    
    # Collect all valid context values across all entities
    all_context_values: Set[str] = set()
    entity_attribute_truth: Dict[Tuple[str, str], str] = {}
    
    for ent_name, attrs in entity_map.items():
        for k, v in attrs.items():
            if k != "name":
                str_val = str(v).strip()
                all_context_values.add(str_val.lower())
                entity_attribute_truth[(ent_name, k)] = str_val

    extracted_claims = extract_triples_from_text(response_text, ground_truth)
    
    correct_bindings = 0
    binding_errors = 0
    hallucinations = 0
    
    evaluated_claims = []

    for ent, attr, val in extracted_claims:
        val_lower = val.lower()
        true_val = entity_attribute_truth.get((ent, attr))
        
        if true_val is not None and true_val.lower() == val_lower:
            # Correct entity + value
            correct_bindings += 1
            evaluated_claims.append({"claim": (ent, attr, val), "status": "correct"})
        elif val_lower in all_context_values:
            # Value is present in context, but bound to WRONG entity! (Binding Error / MEI)
            binding_errors += 1
            evaluated_claims.append({"claim": (ent, attr, val), "status": "binding_error"})
        else:
            # Value absent from context entirely (Hallucination)
            hallucinations += 1
            evaluated_claims.append({"claim": (ent, attr, val), "status": "hallucination"})

    total_bound_claims = correct_bindings + binding_errors
    ber_k = (binding_errors / total_bound_claims) if total_bound_claims > 0 else 0.0

    return {
        "ber_k": ber_k,
        "correct_bindings": correct_bindings,
        "binding_errors": binding_errors,
        "hallucinations": hallucinations,
        "total_referenced_claims": total_bound_claims,
        "evaluated_claims": evaluated_claims
    }
