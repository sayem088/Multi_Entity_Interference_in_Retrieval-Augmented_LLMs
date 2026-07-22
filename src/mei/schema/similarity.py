"""
Schema Similarity Module for MEI Framework.
Computes attribute key set Jaccard similarity across co-retrieved entities.
"""

from typing import Set, List, Dict, Any, Union

def jaccard_similarity(keys1: Union[Set[str], List[str]], keys2: Union[Set[str], List[str]]) -> float:
    """
    Computes Jaccard Similarity between two sets of attribute keys:
    J(A, B) = |A ∩ B| / |A ∪ B|
    Returns 0.0 if both sets are empty.
    """
    s1 = set(keys1)
    s2 = set(keys2)
    
    if not s1 and not s2:
        return 0.0
    
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    
    return intersection / union if union > 0 else 0.0


def pairwise_mean_similarity(entity_schemas: List[Union[Set[str], List[str], Dict[str, Any]]]) -> float:
    """
    Computes average pairwise Jaccard similarity across all pairs of entities in a scenario context.
    If 0 or 1 entity is provided, returns 0.0.
    """
    if len(entity_schemas) <= 1:
        return 0.0
    
    # Standardize input to attribute key sets
    key_sets = []
    for item in entity_schemas:
        if isinstance(item, dict):
            # If item is entity dict, use its keys or attributes subdict
            attrs = item.get("attributes", item)
            key_sets.append(set(attrs.keys()) if isinstance(attrs, dict) else set(attrs))
        else:
            key_sets.append(set(item))
            
    total_sim = 0.0
    num_pairs = 0
    
    n = len(key_sets)
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += jaccard_similarity(key_sets[i], key_sets[j])
            num_pairs += 1
            
    return total_sim / num_pairs if num_pairs > 0 else 0.0


def compute_scenario_schema_similarity(entities: List[Dict[str, Any]]) -> float:
    """
    Extracts attribute keys from a list of entity dictionaries and computes the context-level pairwise mean similarity.
    """
    return pairwise_mean_similarity(entities)
