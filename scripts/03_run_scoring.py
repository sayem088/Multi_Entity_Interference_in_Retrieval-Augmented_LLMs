#!/usr/bin/env python3
"""
Pipeline Step 03: Run Response Scoring.
Reads raw responses from results/stage1/raw_responses/, matches against ground truth,
scores every response on BER_k, EA-F1, and RAGAS Faithfulness, and writes to results/stage1/scored_responses/.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.scoring.ber_k import compute_ber_k
from mei.scoring.ea_f1 import compute_ea_f1
from mei.scoring.ragas_wrapper import compute_ragas_faithfulness

def main():
    print("=== Step 03: Scoring Generated Responses ===")
    
    raw_dir = Path("results/stage1/raw_responses")
    gt_base_dir = Path("data/processed/ground_truth")
    scored_dir = Path("results/stage1/scored_responses")
    scored_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(raw_dir.glob("**/*_raw.json"))
    print(f"Found {len(raw_files)} raw response files to score.")

    for raw_path in raw_files:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_record = json.load(f)

        scenario_id = raw_record["scenario_id"]
        domain_id = raw_record["domain_id"]
        response_text = raw_record["response"]

        # Load matching ground truth record
        gt_path = gt_base_dir / domain_id / f"{scenario_id}_gt.json"
        if not gt_path.exists():
            print(f"Warning: Ground truth file missing for {scenario_id}")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        # Compute all 3 metrics
        ber_results = compute_ber_k(response_text, ground_truth)
        ea_results = compute_ea_f1(response_text, ground_truth)
        ragas_results = compute_ragas_faithfulness(response_text, ground_truth)

        # Combine into scored record
        scored_record = dict(raw_record)
        scored_record.update(ber_results)
        scored_record.update(ea_results)
        scored_record.update(ragas_results)

        dom_scored_dir = scored_dir / domain_id
        dom_scored_dir.mkdir(parents=True, exist_ok=True)

        out_path = dom_scored_dir / f"{scenario_id}_scored.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scored_record, f, indent=2)

    print("Step 03 Finished Successfully!\n")

if __name__ == "__main__":
    main()
