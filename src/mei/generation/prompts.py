"""
Prompt templates for Stage 1 Generation.
Uses a fixed comparison prompt template across all 2x2 cells to avoid prompt confounding.
"""

SYSTEM_PROMPT = (
    "You are an accurate research assistant. Your task is to provide factual comparison "
    "summaries strictly grounded in the retrieved context provided."
)

FORMAT_COMPARISON_PROMPT = """Below is retrieved information for {entity_count} entities:

{context_text}

Task: Provide a detailed comparative summary of all the retrieved entities listed above.
For each entity, clearly state its name and explicitly compare its attributes (such as capacity, established year, location, budget, contact details, or opening hours). Ensure every attribute value is bound to the exact entity it belongs to.
"""

def build_context_string(entities: list) -> str:
    """Formats a list of entity dictionaries into a standardized context block."""
    blocks = []
    for idx, ent in enumerate(entities, 1):
        lines = [f"Entity {idx}: {ent.get('name', 'Unknown')}"]
        lines.append(f"Category: {ent.get('category', 'N/A')}")
        lines.append("Attributes:")
        attrs = ent.get("attributes", {})
        for k, v in attrs.items():
            if k != "name":
                lines.append(f"  - {k}: {v}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
