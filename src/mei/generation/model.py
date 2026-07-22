"""
Model Generation Interface for Qwen2.5-7B-Instruct with Local Fallback & Offline Folder Support.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional

class LLMGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        configured_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        
        # Check if local model folder exists inside project (e.g. models/Qwen2.5-7B-Instruct)
        local_model_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Qwen2.5-7B-Instruct"
        if local_model_path.exists() and any(local_model_path.iterdir()):
            print(f"[LLMGenerator] Found local offline model weights at: {local_model_path}")
            self.model_name = str(local_model_path)
        else:
            self.model_name = configured_name

        self.provider = config.get("provider", "fallback")
        self.device = config.get("device", "cpu")
        self.temperature = config.get("temperature", 0.3)
        self.seed = config.get("seed", 42)
        self.max_new_tokens = config.get("max_new_tokens", 300)

        self._model = None
        self._tokenizer = None

        if self.provider == "huggingface":
            self._init_huggingface_model()

    def _init_huggingface_model(self):
        """Initializes PyTorch Hugging Face Transformers model (for University GPU execution)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"[LLMGenerator] Loading {self.model_name} on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print("[LLMGenerator] Model successfully loaded.")
        except Exception as e:
            print(f"[LLMGenerator Warning] Failed to load PyTorch CUDA model: {e}")
            print("[LLMGenerator Fallback] Reverting provider to local fallback mode.")
            self.provider = "fallback"

    def generate(self, prompt: str, scenario_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a model completion.
        If provider is 'huggingface', runs full CUDA inference.
        If provider is 'fallback', generates structured completions for local testing.
        """
        if self.provider == "huggingface" and self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0
                )
            response = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        return self._generate_fallback(prompt, scenario_data)

    def _generate_fallback(self, prompt: str, scenario_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a structured comparative response for local development.
        Simulates subtle attribute binding interference for high-similarity cells.
        """
        if not scenario_data or "entities" not in scenario_data:
            return "Local fallback generator: Comparison generated successfully."

        entities = scenario_data["entities"]
        cell_label = scenario_data.get("cell_label", "")
        sim_type = scenario_data.get("schema_similarity_type", "low")
        
        paragraphs = ["Here is a detailed comparative summary of the entities:"]

        # If high similarity, simulate occasional misbinding (swap values between entities)
        should_swap = (sim_type == "high" and random.random() < 0.40)
        
        processed_attrs = []
        for idx, ent in enumerate(entities):
            name = ent.get("name", f"Entity {idx+1}")
            attrs = ent.get("attributes", {})
            
            summary_parts = [f"**{name}**"]
            for k, v in attrs.items():
                if k == "name":
                    continue
                val_to_use = v
                # Simulate a binding error swap if in high similarity cell
                if should_swap and idx > 0 and k in entities[0].get("attributes", {}):
                    if random.random() < 0.5:
                        val_to_use = entities[0]["attributes"][k] # Swap with first entity's value
                        
                summary_parts.append(f"has {k} as {val_to_use}")
            
            paragraphs.append(", ".join(summary_parts) + ".")

        return "\n\n".join(paragraphs)
