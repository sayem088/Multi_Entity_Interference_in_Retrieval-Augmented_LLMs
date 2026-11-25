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

def _find_entity_for_line(line: str, entity_names: List[str]) -> str:
    """Finds the unambiguous entity that this text line belongs to."""
    clean_line = re.sub(r'[^\w\s]', ' ', line.lower())
    matched_ent = None

    for ent_name in entity_names:
        clean_ent = re.sub(r'[^\w\s]', ' ', ent_name.lower())
        if clean_ent in clean_line or ent_name.lower() in line.lower():
            if matched_ent is None or len(ent_name) > len(matched_ent):
                matched_ent = ent_name

    return matched_ent


def extract_triples_from_text(response_text: str, ground_truth: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Extracts (entity, attribute, value) claims from response_text matching the known ground_truth schema,
    as well as novel/hallucinated attribute assertions.
    """
    entity_map = ground_truth.get("entity_map", {})
    extracted_claims = []
    entity_names = list(entity_map.keys())

    # Build pool of all known attribute keys and all valid values for each key in this scenario
    all_known_attrs: Dict[str, Set[str]] = {}
    for ent, attrs in entity_map.items():
        for k, v in attrs.items():
            if k not in ["name", "study_or_drug_name"]:
                all_known_attrs.setdefault(k, set()).add(str(v).strip())

    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    for line in lines:
        matched_ent = _find_entity_for_line(line, entity_names)
        if not matched_ent:
            continue

        seen_attrs = set()

        # 1. Check known schema attributes
        for k, possible_vals in all_known_attrs.items():
            key_variants = [k.lower(), k.replace("_", " ").lower(), k.replace("_", "-").lower()]

            key_found = False
            for kv in key_variants:
                if re.search(r'\b' + re.escape(kv) + r'\b', line, re.IGNORECASE):
                    key_found = True
                    break

            if not key_found:
                continue

            sorted_vals = sorted(list(possible_vals), key=len, reverse=True)
            for val in sorted_vals:
                str_val = str(val).strip()
                val_pattern = r'(?:^|[^\w])' + re.escape(str_val) + r'(?:[^\w]|$)'
                if re.search(val_pattern, line, re.IGNORECASE):
                    if k not in seen_attrs:
                        extracted_claims.append((matched_ent, k, str_val))
                        seen_attrs.add(k)
                    break

        # 2. Check for explicit "<attribute> as <value>" or "<attribute>: <value>" patterns (captures hallucinations)
        if all_known_attrs:
            key_regex_parts = []
            for k in all_known_attrs.keys():
                key_regex_parts.append(re.escape(k))
                key_regex_parts.append(re.escape(k.replace("_", " ")))
            attr_pattern = "|".join(key_regex_parts)

            pattern = rf'(?:^|[,\s])(\b(?:{attr_pattern})\b)\s*(?::|\bas\b|=)\s*([^,.\n]+)'
            for m in re.finditer(pattern, line, re.IGNORECASE):
                raw_attr = m.group(1).strip().lower().replace(" ", "_")
                raw_val = m.group(2).strip()

                # Normalize attribute name against known keys
                canonical_attr = None
                for k in all_known_attrs.keys():
                    if raw_attr in [k.lower(), k.replace("_", " ").lower()]:
                        canonical_attr = k
                        break

                if canonical_attr and canonical_attr not in seen_attrs:
                    extracted_claims.append((matched_ent, canonical_attr, raw_val))
                    seen_attrs.add(canonical_attr)

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
