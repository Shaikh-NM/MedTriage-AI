from autogen import AssistantAgent

MEDICAL_KNOWLEDGE_PROMPT = """
You are a medical knowledge assistant.

Rules:
- Map input symptoms to POSSIBLE conditions only — never diagnose definitively
- Identify any red flag symptom combinations
- Return ONLY valid JSON. No prose.
- Never suggest medications or dosages

Output schema:
{
  "possible_conditions": ["list of conditions"],
  "red_flags": ["dangerous symptom combos if any"],
  "knowledge_notes": "brief rationale string"
}
"""

def create_medical_knowledge_agent(llm_config: dict) -> AssistantAgent:
    return AssistantAgent(
        name="MedicalKnowledgeAgent",
        system_message=MEDICAL_KNOWLEDGE_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )