#!/usr/bin/env python3
"""
Pipeline Step 01: Build Stimulus Set.
Constructs 2x2 factorial stimulus datasets for Domain 1 (OSM POI) and Domain 2 (Wiki Infoboxes).
"""

import sys
import yaml
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.stimulus.build_stimulus_set import StimulusBuilder

def main():
    print("=== Step 01: Building Controlled Stimulus Sets ===")
    
    # Load configs
    with open("configs/domains.yaml", "r", encoding="utf-8") as f:
        domains_config = yaml.safe_load(f)["domains"]
        
    with open("configs/stimulus_design.yaml", "r", encoding="utf-8") as f:
        stim_config = yaml.safe_load(f)

    cells = stim_config["cells"]
    target_n = stim_config.get("target_n_per_cell", 40)
    seed = stim_config.get("seed", 42)

    builder = StimulusBuilder(seed=seed)

    for dom in domains_config:
        domain_id = dom["id"]
        categories = dom["categories"]
        print(f"Building stimulus set for domain: {domain_id} (target={target_n} per cell)...")
        
        scenarios = builder.build_domain_stimulus_set(
            domain_id=domain_id,
            categories=categories,
            cells_config=cells,
            target_n_per_cell=target_n
        )
        print(f"  Completed {len(scenarios)} scenarios for {domain_id}.")

    print("Step 01 Finished Successfully!\n")

if __name__ == "__main__":
    main()
