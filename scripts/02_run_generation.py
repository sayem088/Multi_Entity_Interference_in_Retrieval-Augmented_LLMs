#!/usr/bin/env python3
"""
Pipeline Step 02: Run Model Generation.
Loads stimulus scenarios, applies fixed comparison template, calls Qwen2.5-7B-Instruct (or fallback),
and writes raw responses to results/stage1/raw_responses/.
"""

import sys
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.generation.model import LLMGenerator
from mei.generation.prompts import FORMAT_COMPARISON_PROMPT, build_context_string

def main():
    print("=== Step 02: Running Model Generation ===")
    
    with open("configs/model.yaml", "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    generator = LLMGenerator(model_config)
    print(f"Generator initialized with provider='{generator.provider}', device='{generator.device}'")

    stimulus_dir = Path("data/processed/stimulus_sets")
    out_dir = Path("results/stage1/raw_responses")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = list(stimulus_dir.glob("**/*.json"))
    print(f"Found {len(scenario_files)} scenario files for generation.")

    for s_path in scenario_files:
        with open(s_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        scenario_id = scenario["scenario_id"]
        domain_id = scenario["domain_id"]
        entities = scenario["entities"]

        context_str = build_context_string(entities)
        prompt = FORMAT_COMPARISON_PROMPT.format(
            entity_count=len(entities),
            context_text=context_str
        )

        response_text = generator.generate(prompt, scenario_data=scenario)

        # Output raw response record
        record = {
            "scenario_id": scenario_id,
            "domain_id": domain_id,
            "cell_label": scenario["cell_label"],
            "schema_similarity_type": scenario["schema_similarity_type"],
            "schema_similarity_score": scenario["schema_similarity_score"],
            "entity_count": scenario["entity_count"],
            "prompt": prompt,
            "response": response_text,
            "entities": entities
        }

        domain_out_dir = out_dir / domain_id
        domain_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = domain_out_dir / f"{scenario_id}_raw.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    print("Step 02 Finished Successfully!\n")

if __name__ == "__main__":
    main()
