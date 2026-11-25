#!/usr/bin/env python3
"""
2x2 Factorial Stimulus Sets & Ground Truth Generator.
Constructs balanced test scenarios across Domain 1 (GeoAI) and Domain 2 (Healthcare).
Factors:
  - Factor 1: Schema Similarity (Low vs High)
  - Factor 2: Entity Count (k=2 vs k=8)

Generates:
  1. `data/processed/stimulus_sets/`
  2. `data/processed/ground_truth/`
"""

import os
import json
import random
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

GEO_CLEAN_FILE = PROCESSED_DATA_DIR / "geoai_clean.json"
HEALTH_CLEAN_FILE = PROCESSED_DATA_DIR / "healthcare_clean.json"

STIMULUS_BASE_DIR = PROCESSED_DATA_DIR / "stimulus_sets"
GT_BASE_DIR = PROCESSED_DATA_DIR / "ground_truth"

GEO_STIM_DIR = STIMULUS_BASE_DIR / "domain1_geo"
HEALTH_STIM_DIR = STIMULUS_BASE_DIR / "domain2_health"

GEO_GT_DIR = GT_BASE_DIR / "domain1_geo"
HEALTH_GT_DIR = GT_BASE_DIR / "domain2_health"

for p in [GEO_STIM_DIR, HEALTH_STIM_DIR, GEO_GT_DIR, HEALTH_GT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for scientific reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_N_PER_CELL = 25  # 25 scenarios per cell x 4 cells x 2 domains = 200 total scenarios

# 2x2 Factorial Cell Definitions
CELL_CONFIGS = [
    {
        "cell_id": "cell1_low_sim_low_count",
        "cell_label": "low_sim_low_count",
        "similarity_type": "low",
        "count": 2,
        "description": "2 entities from distinct categories with disjoint attribute schemas."
    },
    {
        "cell_id": "cell2_high_sim_low_count",
        "cell_label": "high_sim_low_count",
        "similarity_type": "high",
        "count": 2,
        "description": "2 entities from the same category with identical attribute schemas."
    },
    {
        "cell_id": "cell3_low_sim_high_count",
        "cell_label": "low_sim_high_count",
        "similarity_type": "low",
        "count": 8,
        "description": "8 entities from mixed distinct categories with low schema overlap."
    },
    {
        "cell_id": "cell4_high_sim_high_count",
        "cell_label": "high_sim_high_count",
        "similarity_type": "high",
        "count": 8,
        "description": "8 entities from the same category with identical attribute schemas."
    }
]

def compute_jaccard_similarity(entities: list) -> float:
    """Computes mean pairwise Jaccard similarity between entity attribute key sets."""
    if len(entities) < 2:
        return 1.0
    key_sets = [set(ent["attributes"].keys()) for ent in entities]
    pairs = 0
    total_jaccard = 0.0
    for i in range(len(key_sets)):
        for j in range(i + 1, len(key_sets)):
            s1, s2 = key_sets[i], key_sets[j]
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            jaccard = (intersection / union) if union > 0 else 0.0
            total_jaccard += jaccard
            pairs += 1
    return round(total_jaccard / pairs, 4) if pairs > 0 else 1.0

def flatten_geo_attributes(ent: dict) -> dict:
    """Combines spatial and service attributes into a single attribute mapping."""
    attrs = {"name": ent["name"], "category": ent["category"]}
    attrs.update(ent.get("spatial_attributes", {}))
    attrs.update(ent.get("service_attributes", {}))
    return attrs

def flatten_health_attributes(ent: dict) -> dict:
    """Combines protocol and clinical attributes into a single attribute mapping."""
    attrs = {"name": ent["name"], "category": ent["category"], "nct_id": ent["nct_id"]}
    attrs.update(ent.get("protocol_attributes", {}))
    return attrs

def generate_domain_scenarios(domain_id: str, domain_name: str, clean_data: dict, flatten_fn, stim_out_dir: Path, gt_out_dir: Path):
    print(f"\n[*] Generating 2x2 Factorial Scenarios for: {domain_name} ({domain_id})...")
    categories = list(clean_data.keys())
    primary_cat = categories[0]  # hospital for GeoAI, oncology_trial for Healthcare
    other_cats = categories[1:]

    total_scenarios_generated = 0

    for cell in CELL_CONFIGS:
        cell_id = cell["cell_id"]
        cell_label = cell["cell_label"]
        sim_type = cell["similarity_type"]
        k = cell["count"]
        desc = cell["description"]

        print(f"    -> Building Cell: {cell_label} (k={k}, Sim={sim_type.upper()})...")

        for idx in range(1, TARGET_N_PER_CELL + 1):
            sampled_entities = []

            if sim_type == "high":
                # Sample k entities from the primary homogeneous category (hospital or oncology_trial)
                pool = clean_data[primary_cat]
                chosen = random.sample(pool, k)
                for ent in chosen:
                    sampled_entities.append({
                        "entity_id": ent["entity_id"],
                        "category": ent["category"],
                        "name": ent["name"],
                        "attributes": flatten_fn(ent)
                    })
            else:
                # Low similarity: sample from distinct heterogeneous categories
                if k == 2:
                    # 1 primary + 1 secondary from other categories
                    chosen_cat2 = random.choice(other_cats)
                    ent1 = random.choice(clean_data[primary_cat])
                    ent2 = random.choice(clean_data[chosen_cat2])
                    chosen = [ent1, ent2]
                else:
                    # k=8: exactly 1 entity from each of the 8 distinct categories
                    chosen = [random.choice(clean_data[c]) for c in categories[:8]]

                for ent in chosen:
                    sampled_entities.append({
                        "entity_id": ent["entity_id"],
                        "category": ent["category"],
                        "name": ent["name"],
                        "attributes": flatten_fn(ent)
                    })

            # Compute actual schema similarity score for this scenario
            jaccard_score = compute_jaccard_similarity(sampled_entities)
            scenario_name = f"{domain_id}_{cell_id}_{idx:03d}"

            # 1. Stimulus Record (Exam Paper)
            stimulus_record = {
                "scenario_id": scenario_name,
                "domain_id": domain_id,
                "domain_name": domain_name,
                "cell_id": cell_id,
                "cell_label": cell_label,
                "cell_description": desc,
                "schema_similarity_type": sim_type,
                "schema_similarity_score": jaccard_score,
                "entity_count": k,
                "entities": sampled_entities
            }

            # 2. Ground Truth Record (Answer Key)
            ground_truth_record = {
                "scenario_id": scenario_name,
                "domain_id": domain_id,
                "cell_label": cell_label,
                "schema_similarity_score": jaccard_score,
                "entity_count": k,
                "entity_map": {
                    ent["name"]: ent["attributes"] for ent in sampled_entities
                }
            }

            # Save files
            stim_path = stim_out_dir / f"{scenario_name}.json"
            gt_path = gt_out_dir / f"{scenario_name}_gt.json"

            with open(stim_path, "w", encoding="utf-8") as f:
                json.dump(stimulus_record, f, indent=2, ensure_ascii=False)

            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(ground_truth_record, f, indent=2, ensure_ascii=False)

            total_scenarios_generated += 1

        print(f"       [DONE] Generated {TARGET_N_PER_CELL} scenarios for {cell_label}.")

    print(f"[*] Total Scenarios Generated for {domain_id}: {total_scenarios_generated}")

def main():
    print("=" * 80)
    print("STARTING 2x2 FACTORIAL STIMULUS & GROUND TRUTH GENERATION")
    print(f"Target Scenarios: {TARGET_N_PER_CELL} per cell x 4 cells x 2 domains = {TARGET_N_PER_CELL * 4 * 2} scenarios")
    print("=" * 80)

    # 1. Load Clean Data
    with open(GEO_CLEAN_FILE, "r", encoding="utf-8") as f:
        clean_geo = json.load(f)

    with open(HEALTH_CLEAN_FILE, "r", encoding="utf-8") as f:
        clean_health = json.load(f)

    # 2. Generate Domain 1 (GeoAI)
    generate_domain_scenarios(
        domain_id="domain1_geo",
        domain_name="GeoAI & Spatial Infrastructure (London)",
        clean_data=clean_geo,
        flatten_fn=flatten_geo_attributes,
        stim_out_dir=GEO_STIM_DIR,
        gt_out_dir=GEO_GT_DIR
    )

    # 3. Generate Domain 2 (Healthcare)
    generate_domain_scenarios(
        domain_id="domain2_health",
        domain_name="Healthcare & Biomedical Clinical Protocols",
        clean_data=clean_health,
        flatten_fn=flatten_health_attributes,
        stim_out_dir=HEALTH_STIM_DIR,
        gt_out_dir=HEALTH_GT_DIR
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL FACTORIAL SCENARIOS AND GROUND TRUTH FILES GENERATED!")
    print(f"Stimulus Base Directory:     {STIMULUS_BASE_DIR}")
    print(f"Ground Truth Base Directory: {GT_BASE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
