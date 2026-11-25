"""
=============================================================================
Model Inference Script 1: Qwen2.5-7B-Instruct (GPU / University Environment)
Multi-Entity Interference in Retrieval-Augmented Language Models
=============================================================================
This script runs inference using the 7B parameter model (Qwen2.5-7B-Instruct).
It reads stimuli from:
  step-2-simulation-exam-ans-script/stimulus_sets/
And writes structured raw model responses to:
  step-3-apply-to-model/raw_model_responses/qwen_7b/

Usage:
  # Normal execution on GPU (University Lab / Colab):
  python run_inference_7b.py --device cuda

  # Quick test on first 2 scenarios per domain:
  python run_inference_7b.py --device cuda --limit 2

  # Pipeline dry-run with simulated responses (zero GPU/CPU cost):
  python run_inference_7b.py --mode swap --limit 5
=============================================================================
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

# Paths relative to repository root
ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR = "Qwen/Qwen2.5-7B-Instruct"
STIMULUS_BASE_DIR = ROOT_DIR / "data" / "processed" / "stimulus_sets"
RESPONSE_BASE_DIR = ROOT_DIR / "experiments" / "qwen2_5_7b" / "raw_responses"

DOMAINS = ["domain1_geo", "domain2_health"]


def format_rag_prompt(scenario: dict) -> str:
    """Builds a standardized zero-shot RAG information extraction prompt."""
    domain_id = scenario.get("domain_id", "")
    entities = scenario.get("entities", [])
    
    if "geo" in domain_id:
        domain_desc = "Central London GeoAI & Spatial Infrastructure"
    else:
        domain_desc = "Clinical Trial Protocols & Healthcare Interventions"

    context_lines = []
    for ent in entities:
        ent_name = ent.get("name", "Unknown Entity")
        cat = ent.get("category", "")
        context_lines.append(f"Entity: {ent_name} (Category: {cat})")
        for k, v in ent.get("attributes", {}).items():
            if k not in ["name", "category"]:
                context_lines.append(f"  - {k}: {v}")
        context_lines.append("")

    context_str = "\n".join(context_lines).strip()

    prompt = f"""You are a precise, factual information extraction assistant specializing in {domain_desc}.
Read the retrieved context below carefully. Answer the question relying ONLY on the explicit facts provided.
CRITICAL INSTRUCTION: Do NOT confuse, mix, or interchange attributes between different entities. Every attribute must remain strictly bound to its correct entity.

[RETRIEVED CONTEXT]
{context_str}

[QUESTION]
For every entity listed above, output its exact name and all of its associated attributes in the exact format:
Entity: <Entity Name>
- <attribute_name>: <attribute_value>
...
"""
    return prompt.strip()


def simulate_model_response(scenario: dict, mode: str) -> str:
    """
    Fast deterministic dry-run generator for testing the downstream scoring pipeline.
    - 'clean': 100% correct attribute binding (BER = 0).
    - 'swap': Injects deterministic attribute cross-talk between entities to simulate
              multi-entity binding interference (BER > 0).
    """
    entities = scenario.get("entities", [])
    if not entities:
        return "No entities found."

    lines = []
    if mode == "clean" or len(entities) == 1:
        for ent in entities:
            lines.append(f"Entity: {ent['name']}")
            for k, v in ent.get("attributes", {}).items():
                if k not in ["name", "category"]:
                    lines.append(f"- {k}: {v}")
            lines.append("")
        return "\n".join(lines).strip()

    # mode == 'swap' (simulate binding cross-talk)
    ent_names = [e["name"] for e in entities]
    all_attr_keys = list(entities[0].get("attributes", {}).keys())
    eligible_keys = [k for k in all_attr_keys if k not in ["name", "category"]]

    for i, ent in enumerate(entities):
        lines.append(f"Entity: {ent['name']}")
        attrs = dict(ent.get("attributes", {}))

        # In swap mode, exchange 1 shared attribute with the adjacent entity
        if eligible_keys:
            swap_key = eligible_keys[i % len(eligible_keys)]
            neighbor = entities[(i + 1) % len(entities)]
            neighbor_val = neighbor.get("attributes", {}).get(swap_key, attrs.get(swap_key))
            attrs[swap_key] = neighbor_val

        for k, v in attrs.items():
            if k not in ["name", "category"]:
                lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines).strip()


def run_inference_live(model, tokenizer, prompt: str, device: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Runs actual autoregressive generation with Qwen2.5-7B-Instruct."""
    import torch

    messages = [
        {"role": "system", "content": "You are a precise, factual information extraction assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the generated tokens (strip input tokens)
    input_len = model_inputs.input_ids.shape[1]
    response_tokens = generated_ids[0][input_len:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    return response_text


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-7B-Instruct Inference (7B Model)")
    parser.add_argument("--domain", choices=["domain1_geo", "domain2_health", "all"], default="all",
                        help="Which domain to evaluate (default: all)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Compute device (default: cuda for GPU cluster)")
    parser.add_argument("--mode", choices=["live", "swap", "clean"], default="live",
                        help="Inference mode: 'live' (real 7B model), 'swap' (simulated cross-talk), 'clean' (simulated perfect)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of scenarios per domain for quick testing")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_DIR),
                        help="Path to Qwen2.5-7B-Instruct weights")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1 for factual RAG)")
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Entity RAG Interference Benchmark - Model 1 (Qwen2.5-7B-Instruct)")
    print("=" * 70)
    print(f"[*] Mode       : {args.mode}")
    print(f"[*] Device     : {args.device}")
    print(f"[*] Model Path : {args.model_path}")
    print(f"[*] Output Dir : {RESPONSE_BASE_DIR}")
    print(f"[*] Domains    : {args.domain}")
    print(f"[*] Limit      : {args.limit or 'All 100 per domain'}")
    print("=" * 70)

    # Load model if live mode
    model = None
    tokenizer = None
    if args.mode == "live":
        print(f"[*] Loading Qwen2.5-7B-Instruct onto {args.device.upper()}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto" if args.device == "cuda" else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            if args.device == "cpu":
                model = model.to("cpu")
            model.eval()
            print("[+] Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"[!] ERROR loading 7B model: {e}")
            print("[!] If running on CPU laptop, consider using 'run_inference_tinyllama.py' or '--mode swap'.")
            sys.exit(1)

    domains_to_process = DOMAINS if args.domain == "all" else [args.domain]
    total_processed = 0

    for dom in domains_to_process:
        stim_dir = STIMULUS_BASE_DIR / dom
        out_dir = RESPONSE_BASE_DIR / dom
        out_dir.mkdir(parents=True, exist_ok=True)

        if not stim_dir.exists():
            print(f"[!] Warning: Stimulus directory {stim_dir} not found. Skipping.")
            continue

        files = sorted(list(stim_dir.glob("*.json")))
        if args.limit:
            files = files[:args.limit]

        print(f"\n---> Processing {dom} ({len(files)} scenarios)...")

        for idx, fpath in enumerate(files, 1):
            with open(fpath, "r", encoding="utf-8") as f:
                scenario = json.load(f)

            scen_id = scenario.get("scenario_id", fpath.stem)
            out_file = out_dir / f"{scen_id}_response.json"

            prompt = format_rag_prompt(scenario)

            start_t = time.time()
            if args.mode == "live":
                raw_resp = run_inference_live(model, tokenizer, prompt, args.device, args.max_tokens, args.temperature)
            else:
                raw_resp = simulate_model_response(scenario, args.mode)
            elapsed_sec = round(time.time() - start_t, 3)

            result_payload = {
                "scenario_id": scen_id,
                "domain_id": scenario.get("domain_id", dom),
                "cell_id": scenario.get("cell_id", ""),
                "cell_label": scenario.get("cell_label", ""),
                "schema_similarity_score": scenario.get("schema_similarity_score", 0.0),
                "entity_count": scenario.get("entity_count", 0),
                "model_name": "Qwen2.5-7B-Instruct",
                "mode": args.mode,
                "device": args.device,
                "prompt": prompt,
                "raw_response": raw_resp,
                "generation_time_sec": elapsed_sec,
                "timestamp": datetime.now().isoformat()
            }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2, ensure_ascii=False)

            total_processed += 1
            print(f"  [{idx:03d}/{len(files):03d}] {scen_id} -> Done ({elapsed_sec}s)")

    print("\n" + "=" * 70)
    print(f"[+] All done! Processed {total_processed} scenarios.")
    print(f"[+] Responses saved to: {RESPONSE_BASE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
