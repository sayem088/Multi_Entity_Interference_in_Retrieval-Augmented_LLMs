"""
=============================================================================
Step 4: Deterministic Scoring and Audit Engine
Multi-Entity Interference in Retrieval-Augmented Language Models
=============================================================================
This script evaluates LLM responses against canonical ground-truth answer keys
without relying on any LLM-as-judge (100% deterministic, reproducible Python audit).

Metrics Computed:
  1. BER_k (Binding Error Rate): Transposition/swap error rate isolating cross-talk.
  2. EA-F1 (Entity-Attributed F1): Precision, Recall, and F1 per Lin et al. (MEBench 2025).
  3. RAGAS Faithfulness: Context token support proxy demonstrating standard evaluation blindness.
  4. Blindness Gap: RAGAS Faithfulness - (1 - BER_k).

Usage:
  # Score TinyLlama responses (laptop):
  python score_and_audit.py --model tinyllama

  # Score Qwen 7B responses (university GPU):
  python score_and_audit.py --model qwen_7b

  # Score all available models:
  python score_and_audit.py --model all
=============================================================================
"""

import os
import re
import csv
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent

STEP2_GT_DIR = ROOT_DIR / "data" / "processed" / "ground_truth"
STEP3_RESP_DIR = ROOT_DIR / "experiments" / "qwen2_5_7b" / "raw_responses"

SCORED_RESP_DIR = ROOT_DIR / "experiments" / "qwen2_5_7b" / "scored_responses"
SUMMARY_REPORTS_DIR = ROOT_DIR / "experiments" / "qwen2_5_7b"

DOMAINS = ["domain1_geo", "domain2_health"]
MODELS = ["tinyllama", "qwen_7b"]


def normalize_key(k: str) -> str:
    """Normalizes an attribute key to lower snake_case."""
    k = k.strip().lower()
    k = re.sub(r'[\"\'\`\:\-]+', ' ', k)
    k = re.sub(r'[\s_]+', '_', k)
    return k.strip('_')


def normalize_val(v: str) -> str:
    """Normalizes an attribute value for exact matching across formats."""
    v = v.strip().lower()
    # Strip wrapping quotes, trailing punctuation
    v = v.strip('\"\'`.,;:')
    # Normalize underscores and hyphens to spaces to handle natural language variations
    v = re.sub(r'[\s_\-]+', ' ', v)
    return v.strip()


def parse_llm_response(text: str, expected_entities: list) -> dict:
    """
    Deterministically parses an LLM response into an entity-attribute dictionary.
    Handles multiple LLM output formats (bullet lists, Markdown bolding, quotes, etc.).
    """
    extracted = {e: {} for e in expected_entities}
    current_entity = None
    lines = text.split("\n")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # 1. Check if this line introduces an entity
        matched_entity = None
        for ent in expected_entities:
            # Check direct containment (case-insensitive)
            if ent.lower() in line.lower():
                matched_entity = ent
                break
            # Check NCT ID match for healthcare trials
            if "NCT" in ent:
                nct_id = ent.split(":")[0].strip()
                if nct_id.lower() in line.lower():
                    matched_entity = ent
                    break

        if matched_entity:
            current_entity = matched_entity
            continue

        # 2. Parse key-value attribute pair under the current entity
        if current_entity and (":" in line):
            # Clean leading bullet markers (-, *, numbers)
            cleaned = re.sub(r'^[\*\-\d\.\s\>\#]+', '', line).strip()
            if ":" in cleaned:
                parts = cleaned.split(":", 1)
                k = normalize_key(parts[0])
                v = normalize_val(parts[1])

                # Exclude trivial non-attribute lines or echoed headers
                if k and v and k not in ["entity", "for_the_entity", "name", "exact_name", "category"]:
                    extracted[current_entity][k] = v

    return extracted


def audit_scenario(scenario_id: str, domain_id: str, response_data: dict, gt_data: dict) -> dict:
    """
    Performs full deterministic audit on a single scenario against its ground truth.
    Categorizes every attribute into: Correct, Swapped (Interference), Omitted, or Hallucinated.
    """
    gt_entity_map = gt_data.get("entity_map", {})
    expected_entities = list(gt_entity_map.keys())
    raw_response = response_data.get("raw_response", "")
    prompt = response_data.get("prompt", "")

    # Parse predictions
    predicted_map = parse_llm_response(raw_response, expected_entities)

    total_gt_attributes = 0
    correct_count = 0
    swap_count = 0
    omitted_count = 0
    hallucinated_count = 0

    detailed_audit = {}

    # Pool of all values across other entities for swap detection
    entity_val_pool = {}
    for ent_name, attrs in gt_entity_map.items():
        entity_val_pool[ent_name] = {
            normalize_key(k): normalize_val(str(v))
            for k, v in attrs.items()
            if normalize_key(k) not in ["name", "category"]
        }

    # Evaluate each expected entity
    for ent_name, true_attrs in gt_entity_map.items():
        detailed_audit[ent_name] = {
            "correct": [],
            "swapped": [],
            "omitted": [],
            "hallucinated": []
        }

        true_norm = {
            normalize_key(k): normalize_val(str(v))
            for k, v in true_attrs.items()
            if normalize_key(k) not in ["name", "category"]
        }
        pred_norm = predicted_map.get(ent_name, {})

        total_gt_attributes += len(true_norm)

        # Check ground truth keys
        for k_true, v_true in true_norm.items():
            if k_true in pred_norm:
                v_pred = pred_norm[k_true]
                if v_pred == v_true:
                    correct_count += 1
                    detailed_audit[ent_name]["correct"].append({"key": k_true, "value": v_pred})
                else:
                    # Check if the predicted value belongs to ANOTHER entity in the context (SWAP)
                    donor_entity = None
                    for other_ent, other_attrs in entity_val_pool.items():
                        if other_ent != ent_name and other_attrs.get(k_true) == v_pred:
                            donor_entity = other_ent
                            break

                    if donor_entity:
                        swap_count += 1
                        detailed_audit[ent_name]["swapped"].append({
                            "key": k_true,
                            "predicted_value": v_pred,
                            "true_value": v_true,
                            "swapped_from_entity": donor_entity
                        })
                    else:
                        hallucinated_count += 1
                        detailed_audit[ent_name]["hallucinated"].append({
                            "key": k_true,
                            "predicted_value": v_pred,
                            "true_value": v_true
                        })
            else:
                omitted_count += 1
                detailed_audit[ent_name]["omitted"].append({"key": k_true, "expected_value": v_true})

        # Check for extra keys in prediction not in GT
        for k_pred, v_pred in pred_norm.items():
            if k_pred not in true_norm:
                hallucinated_count += 1
                detailed_audit[ent_name]["hallucinated"].append({
                    "key": k_pred,
                    "predicted_value": v_pred,
                    "note": "unrecognized_attribute_key"
                })

    # Mathematical Metrics
    total_predicted_pairs = correct_count + swap_count + hallucinated_count

    # 1. BER_k (Binding Error Rate): Proportion of GT attributes corrupted by cross-entity swaps
    ber_k = round(swap_count / total_gt_attributes, 4) if total_gt_attributes > 0 else 0.0

    # 2. Exact Attribute Accuracy
    exact_accuracy = round(correct_count / total_gt_attributes, 4) if total_gt_attributes > 0 else 0.0

    # 3. EA-F1 (Entity-Attributed F1 - MEBench)
    tp = correct_count
    fp = swap_count + hallucinated_count
    fn = omitted_count + swap_count  # A swapped attribute is both FP and FN in standard information extraction

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    ea_f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    # 4. Standard RAG Faithfulness (Token Context Support / RAGAS Blindness Proxy)
    # Checks whether predicted values appear anywhere in the retrieved context string
    prompt_lower = prompt.lower()
    supported_values = 0
    total_checked_values = 0

    for ent_name, preds in predicted_map.items():
        for k, v in preds.items():
            total_checked_values += 1
            if v and (v in prompt_lower):
                supported_values += 1

    ragas_faithfulness = round(supported_values / total_checked_values, 4) if total_checked_values > 0 else 1.0

    # Blindness Gap: High RAGAS despite high Binding Errors
    # Gap = RAGAS Faithfulness - True Bound Recall
    blindness_gap = round(max(0.0, ragas_faithfulness - recall), 4)

    return {
        "scenario_id": scenario_id,
        "domain_id": domain_id,
        "cell_label": gt_data.get("cell_label", ""),
        "schema_similarity_score": gt_data.get("schema_similarity_score", 0.0),
        "entity_count": gt_data.get("entity_count", len(expected_entities)),
        "model_name": response_data.get("model_name", "unknown"),
        "mode": response_data.get("mode", "unknown"),
        "generation_time_sec": response_data.get("generation_time_sec", 0.0),
        "total_gt_attributes": total_gt_attributes,
        "counts": {
            "correct": correct_count,
            "swapped": swap_count,
            "omitted": omitted_count,
            "hallucinated": hallucinated_count,
            "total_predicted": total_predicted_pairs
        },
        "metrics": {
            "BER_k": ber_k,
            "exact_accuracy": exact_accuracy,
            "EA_precision": precision,
            "EA_recall": recall,
            "EA_F1": ea_f1,
            "RAGAS_faithfulness": ragas_faithfulness,
            "blindness_gap": blindness_gap
        },
        "detailed_audit": detailed_audit,
        "timestamp": datetime.now().isoformat()
    }


def score_model(model_name: str, domain_filter: str = "all") -> list:
    """Scores all raw responses for a given model across domains."""
    domains_to_process = DOMAINS if domain_filter == "all" else [domain_filter]
    all_scored_results = []

    print(f"\n======================================================================")
    print(f"Scoring Model Responses: {model_name.upper()}")
    print(f"======================================================================")

    model_resp_dir = STEP3_RESP_DIR if model_name == "qwen_7b" else ROOT_DIR / "experiments" / model_name / "raw_responses"
    if not model_resp_dir.exists():
        print(f"[!] Warning: Response directory {model_resp_dir} does not exist. Skipping.")
        return []

    for dom in domains_to_process:
        dom_resp_dir = model_resp_dir / dom
        dom_gt_dir = STEP2_GT_DIR / dom
        dom_out_dir = SCORED_RESP_DIR / model_name / dom
        dom_out_dir.mkdir(parents=True, exist_ok=True)

        if not dom_resp_dir.exists():
            print(f"[!] Skipping {dom}: Directory {dom_resp_dir} not found.")
            continue

        resp_files = sorted(list(dom_resp_dir.glob("*_response.json")))
        if not resp_files:
            print(f"[*] {dom}: 0 response files found.")
            continue

        print(f"[*] Processing {dom}: Found {len(resp_files)} response files.")

        for fpath in resp_files:
            with open(fpath, "r", encoding="utf-8") as f:
                resp_data = json.load(f)

            scen_id = resp_data.get("scenario_id", fpath.stem.replace("_response", ""))
            gt_file = dom_gt_dir / f"{scen_id}_gt.json"

            if not gt_file.exists():
                print(f"[!] Warning: Ground truth not found for {scen_id} at {gt_file}")
                continue

            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            audit_res = audit_scenario(scen_id, dom, resp_data, gt_data)
            all_scored_results.append(audit_res)

            # Save detailed per-scenario scored audit JSON
            out_scored_file = dom_out_dir / f"{scen_id}_scored.json"
            with open(out_scored_file, "w", encoding="utf-8") as f:
                json.dump(audit_res, f, indent=2, ensure_ascii=False)

            m = audit_res["metrics"]
            print(f"  [+] {scen_id} -> BER_k: {m['BER_k']:.3f} | EA-F1: {m['EA_F1']:.3f} | RAGAS: {m['RAGAS_faithfulness']:.3f} | BlindGap: {m['blindness_gap']:.3f}")

    return all_scored_results


def generate_summary_reports(model_name: str, scored_results: list):
    """Generates structured CSV and JSON summary reports aggregated across factorial cells."""
    if not scored_results:
        print(f"[!] No scored results for {model_name}. Summary skipped.")
        return

    csv_path = SUMMARY_REPORTS_DIR / f"summary_{model_name}.csv"
    json_path = SUMMARY_REPORTS_DIR / f"summary_{model_name}.json"

    # 1. Write CSV
    headers = [
        "scenario_id", "domain_id", "cell_label", "schema_similarity_score",
        "entity_count", "model_name", "mode", "generation_time_sec",
        "total_gt_attributes", "correct", "swapped", "omitted", "hallucinated",
        "BER_k", "exact_accuracy", "EA_precision", "EA_recall", "EA_F1",
        "RAGAS_faithfulness", "blindness_gap"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in scored_results:
            c = r["counts"]
            m = r["metrics"]
            writer.writerow([
                r["scenario_id"], r["domain_id"], r["cell_label"], r["schema_similarity_score"],
                r["entity_count"], r["model_name"], r["mode"], r["generation_time_sec"],
                r["total_gt_attributes"], c["correct"], c["swapped"], c["omitted"], c["hallucinated"],
                m["BER_k"], m["exact_accuracy"], m["EA_precision"], m["EA_recall"], m["EA_F1"],
                m["RAGAS_faithfulness"], m["blindness_gap"]
            ])

    # 2. Write JSON with cell aggregations
    cells = {}
    for r in scored_results:
        cell = r["cell_label"]
        if cell not in cells:
            cells[cell] = {"count": 0, "BER_k": [], "EA_F1": [], "RAGAS": [], "Blindness_Gap": []}
        cells[cell]["count"] += 1
        cells[cell]["BER_k"].append(r["metrics"]["BER_k"])
        cells[cell]["EA_F1"].append(r["metrics"]["EA_F1"])
        cells[cell]["RAGAS"].append(r["metrics"]["RAGAS_faithfulness"])
        cells[cell]["Blindness_Gap"].append(r["metrics"]["blindness_gap"])

    cell_summary = {}
    for cell, vals in cells.items():
        n = vals["count"]
        cell_summary[cell] = {
            "n": n,
            "mean_BER_k": round(sum(vals["BER_k"]) / n, 4),
            "mean_EA_F1": round(sum(vals["EA_F1"]) / n, 4),
            "mean_RAGAS_faithfulness": round(sum(vals["RAGAS"]) / n, 4),
            "mean_blindness_gap": round(sum(vals["Blindness_Gap"]) / n, 4)
        }

    overall_n = len(scored_results)
    overall_summary = {
        "model_name": model_name,
        "total_scenarios_evaluated": overall_n,
        "overall_mean_BER_k": round(sum(r["metrics"]["BER_k"] for r in scored_results) / overall_n, 4),
        "overall_mean_EA_F1": round(sum(r["metrics"]["EA_F1"] for r in scored_results) / overall_n, 4),
        "overall_mean_RAGAS": round(sum(r["metrics"]["RAGAS_faithfulness"] for r in scored_results) / overall_n, 4),
        "overall_mean_blindness_gap": round(sum(r["metrics"]["blindness_gap"] for r in scored_results) / overall_n, 4),
        "factorial_cell_breakdown": cell_summary,
        "generated_at": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY REPORT: {model_name.upper()} (Evaluated {overall_n} Scenarios)")
    print("=" * 70)
    print(f"[*] Overall Mean BER_k        : {overall_summary['overall_mean_BER_k']:.4f}")
    print(f"[*] Overall Mean EA-F1        : {overall_summary['overall_mean_EA_F1']:.4f}")
    print(f"[*] Overall Mean RAGAS        : {overall_summary['overall_mean_RAGAS']:.4f}")
    print(f"[*] Overall Blindness Gap (Δ) : {overall_summary['overall_mean_blindness_gap']:.4f}")
    print("-" * 70)
    print(f"Factorial Cell Breakdown (Cell Label | Mean BER_k | Mean EA-F1 | Blindness Gap):")
    for cell, stat in cell_summary.items():
        print(f"  - {cell:<25} : BER_k={stat['mean_BER_k']:.3f} | EA-F1={stat['mean_EA_F1']:.3f} | BlindGap={stat['mean_blindness_gap']:.3f}")
    print("=" * 70)
    print(f"[+] Detailed CSV saved to : {csv_path}")
    print(f"[+] Summary JSON saved to: {json_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Deterministic Scoring and Audit Engine (Step 4)")
    parser.add_argument("--model", choices=["tinyllama", "qwen_7b", "all"], default="qwen_7b",
                        help="Which model to score (default: tinyllama)")
    parser.add_argument("--domain", choices=["domain1_geo", "domain2_health", "all"], default="all",
                        help="Which domain to score (default: all)")
    args = parser.parse_args()

    models_to_score = MODELS if args.model == "all" else [args.model]

    for mdl in models_to_score:
        results = score_model(mdl, args.domain)
        generate_summary_reports(mdl, results)


if __name__ == "__main__":
    main()
