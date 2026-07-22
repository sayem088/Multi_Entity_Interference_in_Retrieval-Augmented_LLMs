"""
Stimulus Builder Module.
Constructs balanced 2x2 factorial stimulus sets (Low/High Schema Similarity x Low/High Entity Count)
and generates ground-truth mapping references.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from mei.schema.similarity import compute_scenario_schema_similarity
from mei.retrieval.osm_client import OSMClient
from mei.retrieval.wiki_client import WikiClient

class StimulusBuilder:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.osm_client = OSMClient()
        self.wiki_client = WikiClient()

    def build_domain_stimulus_set(
        self,
        domain_id: str,
        categories: List[str],
        cells_config: List[Dict[str, Any]],
        target_n_per_cell: int = 40,
        output_dir: str = "data/processed/stimulus_sets",
        ground_truth_dir: str = "data/processed/ground_truth"
    ) -> List[Dict[str, Any]]:
        """
        Builds stimulus set scenarios across the 4 cells in configs/stimulus_design.yaml.
        """
        domain_output_dir = Path(output_dir) / domain_id
        domain_output_dir.mkdir(parents=True, exist_ok=True)
        
        gt_dir = Path(ground_truth_dir) / domain_id
        gt_dir.mkdir(parents=True, exist_ok=True)

        scenarios = []

        # Pull pools of entities per category
        entity_pool: Dict[str, List[Dict[str, Any]]] = {}
        for cat in categories:
            if "poi" in domain_id or "osm" in domain_id:
                entity_pool[cat] = self.osm_client.fetch_pois_by_category(cat, limit=100)
            else:
                entity_pool[cat] = self.wiki_client.fetch_entities_by_category(cat, limit=100)

        for cell in cells_config:
            cell_label = cell["id"]
            sim_target = cell["similarity"]  # "low" or "high"
            count_target = cell["count"]      # 2 or 8
            
            created_count = 0
            attempts = 0
            max_attempts = 1000

            while created_count < target_n_per_cell and attempts < max_attempts:
                attempts += 1
                
                # Sample entities for high vs low similarity
                if sim_target == "high":
                    # Choose entities from the SAME category (high schema overlap)
                    cat = random.choice(categories)
                    pool = entity_pool.get(cat, [])
                    if len(pool) < count_target:
                        continue
                    sampled_entities = random.sample(pool, count_target)
                else:
                    # Choose entities from DIFFERENT categories (low schema overlap)
                    if len(categories) < count_target:
                        chosen_cats = [random.choice(categories) for _ in range(count_target)]
                    else:
                        chosen_cats = random.sample(categories, count_target)
                    
                    sampled_entities = []
                    for c in chosen_cats:
                        pool = entity_pool.get(c, [])
                        if pool:
                            sampled_entities.append(random.choice(pool))
                    
                    if len(sampled_entities) < count_target:
                        continue

                # Compute exact scenario Jaccard schema similarity
                sim_score = compute_scenario_schema_similarity(sampled_entities)

                # Filter by similarity criteria
                if sim_target == "low" and sim_score > 0.50:
                    continue
                if sim_target == "high" and sim_score < 0.50:
                    continue

                scenario_id = f"{domain_id}_{cell_label}_{created_count + 1:03d}"
                
                scenario = {
                    "scenario_id": scenario_id,
                    "domain_id": domain_id,
                    "cell_label": cell_label,
                    "schema_similarity_type": sim_target,
                    "schema_similarity_score": sim_score,
                    "entity_count": count_target,
                    "entities": sampled_entities
                }

                # Construct ground truth record mapping entity -> attribute -> value
                ground_truth = {
                    "scenario_id": scenario_id,
                    "cell_label": cell_label,
                    "entity_map": {
                        ent["name"]: ent["attributes"] for ent in sampled_entities
                    }
                }

                # Save scenario and ground truth files
                scenario_path = domain_output_dir / f"{scenario_id}.json"
                gt_path = gt_dir / f"{scenario_id}_gt.json"

                with open(scenario_path, "w", encoding="utf-8") as f:
                    json.dump(scenario, f, indent=2)

                with open(gt_path, "w", encoding="utf-8") as f:
                    json.dump(ground_truth, f, indent=2)

                scenarios.append(scenario)
                created_count += 1

        return scenarios
