#!/usr/bin/env python3
"""
Pipeline Step 04: Run Statistical Analysis & Plotting.
Analyzes scored responses from results/stage1/scored_responses/, outputs summary tables,
runs discriminant validity tests, and generates figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mei.analysis.stage1_stats import Stage1Analyzer

def main():
    print("=== Step 04: Running Statistical Analysis & Reporting ===")
    
    analyzer = Stage1Analyzer()
    results = analyzer.run_analysis()

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    discrim = results.get("discriminant_validity", {})
    print(f"Discriminant Validity Check (RQ3):")
    print(f"  Total Scored Responses: {discrim.get('total_responses')}")
    print(f"  RAGAS-Blind MEI Responses: {discrim.get('ragas_blind_mei_count')}")
    print(f"  Rate: {discrim.get('discriminant_validity_rate', 0.0):.2%}")
    print("\nStatistical outputs written to results/stage1/tables/ and figures/.")
    print("Step 04 Finished Successfully!\n")

if __name__ == "__main__":
    main()
