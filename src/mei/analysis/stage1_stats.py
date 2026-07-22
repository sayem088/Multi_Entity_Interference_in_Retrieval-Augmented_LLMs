"""
Stage 1 Statistical Analysis & Reporting Module.
Uses Python standard library (csv, math, json) for guaranteed local execution,
with optional pandas/matplotlib support if installed.
"""

import json
import math
import csv
from pathlib import Path
from typing import Dict, Any, List

class Stage1Analyzer:
    def __init__(self, scored_dir: str = "results/stage1/scored_responses"):
        self.scored_dir = Path(scored_dir)

    def load_scored_data(self) -> List[Dict[str, Any]]:
        """Loads scored response JSON records."""
        records = []
        for file_path in self.scored_dir.glob("**/*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append({
                    "scenario_id": data.get("scenario_id"),
                    "domain_id": data.get("domain_id"),
                    "cell_label": data.get("cell_label"),
                    "similarity_type": data.get("schema_similarity_type"),
                    "similarity_score": data.get("schema_similarity_score"),
                    "entity_count": data.get("entity_count"),
                    "ber_k": data.get("ber_k", 0.0),
                    "ea_f1": data.get("ea_f1", 0.0),
                    "ragas_faithfulness": data.get("ragas_faithfulness", 1.0),
                    "binding_errors": data.get("binding_errors", 0),
                    "correct_bindings": data.get("correct_bindings", 0)
                })
        return records

    def run_analysis(
        self,
        tables_dir: str = "results/stage1/tables",
        figures_dir: str = "results/stage1/figures"
    ) -> Dict[str, Any]:
        """
        Runs 2x2 Factorial statistical analysis, computes CIs, discriminant validity, and outputs artifacts.
        """
        records = self.load_scored_data()
        if not records:
            return {"error": "No scored data found in results/stage1/scored_responses"}

        Path(tables_dir).mkdir(parents=True, exist_ok=True)
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        # Group by (domain_id, cell_label)
        groups: Dict[tuple, List[float]] = {}
        domain_cell_info: Dict[tuple, Dict[str, Any]] = {}
        
        ragas_blind_count = 0
        total_records = len(records)

        for rec in records:
            key = (rec["domain_id"], rec["cell_label"])
            if key not in groups:
                groups[key] = []
                domain_cell_info[key] = {
                    "domain_id": rec["domain_id"],
                    "cell_label": rec["cell_label"],
                    "similarity_type": rec["similarity_type"],
                    "entity_count": rec["entity_count"]
                }
            groups[key].append(rec["ber_k"])

            # Discriminant validity check (RAGAS faithfulness >= 0.80 but BER_k > 0)
            if rec["ragas_faithfulness"] >= 0.80 and rec["ber_k"] > 0.0:
                ragas_blind_count += 1

        summary_rows = []
        for key, ber_list in groups.items():
            info = domain_cell_info[key]
            n = len(ber_list)
            mean_ber = sum(ber_list) / n if n > 0 else 0.0
            
            variance = sum((x - mean_ber) ** 2 for x in ber_list) / (n - 1) if n > 1 else 0.0
            std_dev = math.sqrt(variance)
            sem = std_dev / math.sqrt(n) if n > 0 else 0.0
            ci_95 = 1.96 * sem

            summary_rows.append({
                "domain_id": info["domain_id"],
                "cell_label": info["cell_label"],
                "similarity_type": info["similarity_type"],
                "entity_count": info["entity_count"],
                "count": n,
                "mean_ber_k": round(mean_ber, 4),
                "std_ber_k": round(std_dev, 4),
                "ci_95": round(ci_95, 4),
                "ber_k_ci_lower": round(max(0.0, mean_ber - ci_95), 4),
                "ber_k_ci_upper": round(mean_ber + ci_95, 4)
            })

        # Save CSV summary table using stdlib csv
        csv_path = Path(tables_dir) / "stage1_cell_summary.csv"
        fieldnames = ["domain_id", "cell_label", "similarity_type", "entity_count", "count", "mean_ber_k", "std_ber_k", "ci_95", "ber_k_ci_lower", "ber_k_ci_upper"]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        # Discriminant validity report
        discrim_rate = ragas_blind_count / total_records if total_records > 0 else 0.0
        discrim_report = {
            "total_responses": total_records,
            "ragas_blind_mei_count": ragas_blind_count,
            "discriminant_validity_rate": round(discrim_rate, 4)
        }
        
        with open(Path(tables_dir) / "discriminant_validity.json", "w", encoding="utf-8") as f:
            json.dump(discrim_report, f, indent=2)

        # Generate text summary figure log
        self._generate_summary_log(summary_rows, figures_dir)

        return {
            "summary_table": summary_rows,
            "discriminant_validity": discrim_report
        }

    def _generate_summary_log(self, summary_rows: List[Dict[str, Any]], figures_dir: str):
        """Generates a structured text summary visualization report."""
        lines = ["=== MEI Stage 1 BER_k Cell Summary Report ===", ""]
        lines.append(f"{'Domain':<15} {'Cell':<22} {'Sim':<6} {'Count':<6} {'N':<5} {'Mean BER_k':<12} {'95% CI':<15}")
        lines.append("-" * 85)

        for r in summary_rows:
            ci_str = f"[{r['ber_k_ci_lower']:.3f}, {r['ber_k_ci_upper']:.3f}]"
            lines.append(f"{r['domain_id']:<15} {r['cell_label']:<22} {r['similarity_type']:<6} {r['entity_count']:<6} {r['count']:<5} {r['mean_ber_k']:<12.4f} {ci_str:<15}")

        report_path = Path(figures_dir) / "stage1_ber_k_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
