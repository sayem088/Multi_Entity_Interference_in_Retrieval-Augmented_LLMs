#!/usr/bin/env python3
"""
Deterministic Statistical Verification & ANOVA Reproduction Script.

Reads the 200-scenario Qwen2.5-7B-Instruct evaluation benchmark CSV
(`summary_qwen_7b.csv`), verifies cell & domain aggregations, executes
Two-Way Factorial ANOVA (Type II SS) on both Evaluation Blindness Gap
and Entity-Aware F1, and runs the paired t-test on Blindness Gap.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

EXP_DIR = Path(__file__).resolve().parent
CSV_PATH = EXP_DIR / "summary_qwen_7b.csv"
OUT_JSON = EXP_DIR / "anova_summary.json"


def map_similarity(cell_label: str) -> str:
    return "Low" if "low_sim" in str(cell_label).lower() else "High"


def map_density(cell_label: str, entity_count: int) -> str:
    if "low_count" in str(cell_label).lower() or entity_count == 2:
        return "Low"
    return "High"


def compute_anova(df: pd.DataFrame, dv: str) -> dict:
    formula = f"{dv} ~ C(Factor_Schema_Similarity) + C(Factor_Entity_Density) + C(Factor_Schema_Similarity):C(Factor_Entity_Density)"
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_res = table.loc["Residual", "sum_sq"]
    table["eta_p_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_res)
    table.loc["Residual", "eta_p_sq"] = np.nan

    results = {}
    for idx, row in table.iterrows():
        results[idx] = {
            "sum_sq": round(float(row["sum_sq"]), 6),
            "df": int(row["df"]),
            "F": None if pd.isna(row["F"]) else round(float(row["F"]), 4),
            "p_value": None if pd.isna(row["PR(>F)"]) else float(row["PR(>F)"]),
            "eta_p_sq": None if pd.isna(row["eta_p_sq"]) else round(float(row["eta_p_sq"]), 4),
        }
    return results


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["Factor_Schema_Similarity"] = df["cell_label"].apply(map_similarity)
    df["Factor_Entity_Density"] = df.apply(
        lambda r: map_density(r["cell_label"], r["entity_count"]), axis=1
    )

    print(f"=== Qwen2.5-7B-Instruct Factorial Audit (N = {len(df)}) ===")
    cell_order = [
        "low_sim_low_count",
        "high_sim_low_count",
        "low_sim_high_count",
        "high_sim_high_count",
    ]
    cell_stats = (
        df.groupby("cell_label")[
            ["EA_precision", "EA_recall", "EA_F1", "RAGAS_faithfulness", "blindness_gap", "exact_accuracy"]
        ]
        .agg(["mean", "std"])
        .reindex(cell_order)
    )
    print("\n[1] Factorial Cell Means:")
    for cell in cell_order:
        f1 = cell_stats.loc[cell, ("EA_F1", "mean")]
        rec = cell_stats.loc[cell, ("EA_recall", "mean")]
        ragas = cell_stats.loc[cell, ("RAGAS_faithfulness", "mean")]
        gap = cell_stats.loc[cell, ("blindness_gap", "mean")]
        print(f"  {cell:<22} | EA-F1: {f1:.4f} | Recall: {rec:.4f} | RAGAS: {ragas:.4f} | Gap: {gap:.4f}")

    dom_stats = df.groupby("domain_id")[
        ["EA_F1", "RAGAS_faithfulness", "blindness_gap", "generation_time_sec"]
    ].mean()
    print("\n[2] Domain Means:")
    for dom, row in dom_stats.iterrows():
        print(
            f"  {dom:<15} | EA-F1: {row['EA_F1']:.4f} | RAGAS: {row['RAGAS_faithfulness']:.4f} | "
            f"Gap: {row['blindness_gap']:.4f} | Latency: {row['generation_time_sec']:.2f}s"
        )

    anova_gap = compute_anova(df, "blindness_gap")
    anova_f1 = compute_anova(df, "EA_F1")

    gaps = df["blindness_gap"].dropna()
    t_stat, p_two = stats.ttest_1samp(gaps, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - (p_two / 2)

    print("\n[3] Two-Way Factorial ANOVA on Blindness Gap (Delta_blind):")
    for k, v in anova_gap.items():
        print(f"  {k}: F={v['F']}, p={v['p_value']}, eta_p^2={v['eta_p_sq']}")

    print("\n[4] Two-Way Factorial ANOVA on EA-F1:")
    for k, v in anova_f1.items():
        print(f"  {k}: F={v['F']}, p={v['p_value']}, eta_p^2={v['eta_p_sq']}")

    print(
        f"\n[5] Paired Blindness Test (H_a: Delta_blind > 0): "
        f"mean={gaps.mean():.4f}, std={gaps.std():.4f}, t={t_stat:.4f}, p={p_one:.4e}"
    )

    payload = {
        "model": "Qwen2.5-7B-Instruct",
        "total_scenarios": int(len(df)),
        "anova_blindness_gap": anova_gap,
        "anova_ea_f1": anova_f1,
        "blindness_t_test": {
            "mean_gap": round(float(gaps.mean()), 4),
            "std_gap": round(float(gaps.std()), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value_one_tailed": float(p_one),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved complete statistical report to: {OUT_JSON}")


if __name__ == "__main__":
    main()
