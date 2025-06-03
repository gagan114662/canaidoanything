from creative_loader import load_creative_instructions

# For Your Model Pipeline:
CREATIVE_SYSTEM_PROMPT = """
You are the AI behind the world's most innovative fashion transformation system.

CORE CREATIVE MANDATE:
Transform every input into visual poetry that redefines fashion imagery.

CREATIVE PROCESS:
1. Analyze garment for cultural significance and design elements
2. Generate 5 impossible environments where this garment could exist
3. Select the most emotionally resonant and visually striking scenario
4. Add surreal elements that enhance rather than distract
5. Ensure cultural elements are researched and respectfully integrated

QUALITY GATES:
- Would this concept be featured in the world's top art galleries?
- Does this create an emotional response in viewers?
- Is this technically challenging but achievable?
- Does this respect cultural elements while innovating?

CREATIVE OUTPUT FORMAT:
Always provide:
- Primary concept with impossible/surreal elements
- Cultural context and respectful integration
- Environmental storytelling details
- Technical photography specifications
- Emotional narrative description
"""

# For FLUX Prompt Engineering:
def generate_impossible_scenario() -> str:
    """
    Placeholder function for generating an impossible scenario.
    In a real system, this might involve complex logic or another AI call.
    """
    return "fabric is made of liquid light, and seasons change every hour"

def enhance_creative_prompt(base_garment: str, style_reference: str) -> str:
    creative_enhancer = f"""
    Create a concept that would be worthy of the world's most prestigious fashion exhibitions.

    CREATIVE REQUIREMENTS:
    - Include one impossible element that enhances the story
    - Blend cultural traditions respectfully and innovatively
    - Set in an environment that shouldn't exist but feels real
    - Compose cinematically with emotional depth
    - Show technical mastery that pushes creative boundaries

    BASE: {base_garment}
    STYLE: {style_reference}

    Think like: What would this look like in a world where [{generate_impossible_scenario()}]?
    """

    return creative_enhancer

if __name__ == '__main__':
    print("--- CREATIVE_SYSTEM_PROMPT ---")
    print(CREATIVE_SYSTEM_PROMPT)

    print("\n--- Example of enhanced_creative_prompt ---")
    enhanced_prompt = enhance_creative_prompt("a simple black dress", "surreal photography")
    print(enhanced_prompt)

    print("\n--- Loading instructions using creative_loader ---")
    # For creative concept generation
    # Note: The issue mentions "cultural_creative_fusion.txt" but its content wasn't provided.
    # We'll load the ones we created.
    creative_system_instructions = load_creative_instructions("creative_prompts/revolutionary_creative_ai.txt")
    flux_creative_instructions = load_creative_instructions("creative_prompts/flux_creative_system.txt")
    # cultural_creative_instructions = load_creative_instructions("creative_prompts/cultural_creative_fusion.txt") # This file was not created as content was missing

    print("\n--- Loaded Revolutionary AI Instructions (Placeholder) ---")
    print(creative_system_instructions)
    print("\n--- Loaded FLUX System Instructions (Placeholder) ---")
    print(flux_creative_instructions)
    # print("\n--- Loaded Cultural Creative Fusion Instructions (Placeholder) ---")
    # print(cultural_creative_instructions)
