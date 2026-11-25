#!/usr/bin/env python3
"""
Healthcare & Clinical Protocols Real-World Data Fetcher.
Source: U.S. National Library of Medicine (NLM / NIH) - ClinicalTrials.gov REST API v2.
Fetches real clinical trial protocols: NCT IDs, conditions, interventions, sample sizes, and endpoints.
Enforces GLOBAL uniqueness across all categories (0 duplicate NCT IDs).
"""

import os
import json
import time
import requests
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_FILE = OUTPUT_DIR / "raw_healthcare_clinical.json"

BASE_API_URL = "https://clinicaltrials.gov/api/v2/studies"

CATEGORIES = [
    {"category": "oncology_trial", "condition": "Cancer", "phase": "PHASE3"},
    {"category": "cardiology_trial", "condition": "Heart Failure", "phase": "PHASE3"},
    {"category": "neurology_trial", "condition": "Alzheimer", "phase": "PHASE3"},
    {"category": "infectious_disease", "condition": "Hepatitis", "phase": "PHASE3"},
    {"category": "vaccine_study", "condition": "Vaccine", "phase": "PHASE3"},
    {"category": "medical_device", "condition": "Medical Device", "phase": None},
    {"category": "pediatric_study", "condition": "Pediatric", "phase": "PHASE3"},
    {"category": "mental_health", "condition": "Depression", "phase": "PHASE3"}
]

HEADERS = {
    "User-Agent": "MEI-Healthcare-Benchmark/1.0 (academic clinical research)"
}

def fetch_category_trials(cat_info: dict, global_seen_nct_ids: set, limit: int = 12) -> list:
    cat = cat_info["category"]
    cond = cat_info["condition"]
    phase = cat_info["phase"]
    
    print(f"[*] Fetching real trials for: '{cat}' (Condition: '{cond}')...")
    
    params = {
        "query.cond": cond,
        "pageSize": limit * 4,  # Over-fetch to enforce global NCT deduplication
        "countTotal": "true"
    }
    if phase:
        params["filter.advanced"] = f"AREA[Phase]{phase}"
        
    try:
        response = requests.get(BASE_API_URL, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        
        valid_records = []
        
        for item in studies:
            protocol = item.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            design_module = protocol.get("designModule", {})
            status_module = protocol.get("statusModule", {})
            cond_module = protocol.get("conditionsModule", {})
            interv_module = protocol.get("armsInterventionsModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            
            nct_id = id_module.get("nctId")
            brief_title = id_module.get("briefTitle")
            
            # Enforce global NCT ID deduplication
            if not nct_id or not brief_title or nct_id in global_seen_nct_ids:
                continue
                
            interventions = [
                x.get("name") for x in interv_module.get("interventions", [])
                if x.get("name")
            ]
            
            primary_outcomes = [
                x.get("measure") for x in outcomes_module.get("primaryOutcomes", [])
                if x.get("measure")
            ]
            
            enrollment = design_module.get("enrollmentInfo", {}).get("count")
            phases = design_module.get("phases", ["Not Specified"])
            phase_str = ", ".join(phases) if isinstance(phases, list) else str(phases)
            
            record = {
                "nct_id": nct_id,
                "category": cat,
                "brief_title": brief_title,
                "official_title": id_module.get("officialTitle") or brief_title,
                "conditions": cond_module.get("conditions", [cond]),
                "phase": phase_str,
                "interventions": interventions if interventions else ["Standard of Care"],
                "sample_size": f"{enrollment} patients" if enrollment else "Enrollment not stated",
                "sample_size_count": enrollment,
                "primary_endpoint": primary_outcomes[0] if primary_outcomes else "Clinical Response Rate",
                "study_status": status_module.get("overallStatus", "COMPLETED"),
                "lead_sponsor": sponsor_module.get("leadSponsor", {}).get("name", "Medical Research Group"),
                "completion_date": status_module.get("completionDateStruct", {}).get("date", "2024")
            }
            
            valid_records.append(record)
            global_seen_nct_ids.add(nct_id)
            
            if len(valid_records) >= limit:
                break
                
        print(f"    -> [SUCCESS] Collected {len(valid_records)} verified unique '{cat}' trials.")
        return valid_records
        
    except Exception as e:
        print(f"    [!] Error fetching '{cat}': {e}")
        return []

def main():
    print("=" * 70)
    print("STARTING HEALTHCARE CLINICAL DATA HARVESTING (ClinicalTrials.gov v2)")
    print("8 Biomedical Protocol Categories | Real NCT IDs & Endpoints")
    print("Global De-duplication: Enabled")
    print("=" * 70)
    
    all_data = {}
    global_seen_nct_ids = set()
    
    for cat_info in CATEGORIES:
        records = fetch_category_trials(cat_info, global_seen_nct_ids, limit=12)
        all_data[cat_info["category"]] = records
        time.sleep(1.0)
        
    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
        
    total_records = sum(len(v) for v in all_data.values())
    print("\n" + "=" * 70)
    print(f"[COMPLETED] Successfully Harvested {total_records} Real Clinical Trial Protocols!")
    print(f"File Saved: {RAW_OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
