import json
from autogen import AssistantAgent

MEDICAL_KNOWLEDGE_PROMPT = """
You are a medical knowledge assistant.

Rules:
- Map input symptoms to POSSIBLE conditions only — never diagnose definitively
- Identify any red flag symptom combinations
- Return ONLY valid JSON. No prose, no markdown.
- Never suggest medications or dosages

Output schema:
{
  "possible_conditions": ["list of conditions"],
  "red_flags": ["dangerous symptom combinations if any, empty list if none"],
  "knowledge_notes": "brief rationale string"
}
"""

MEDICAL_KNOWLEDGE_WITH_RAG_PROMPT = """
You are a medical knowledge assistant with access to WHO medical guidelines.

Rules:
- Use the provided WHO reference material to guide your analysis
- Map input symptoms to POSSIBLE conditions only — never diagnose definitively
- Identify any red flag symptom combinations
- Return ONLY valid JSON. No prose, no markdown.
- Never suggest medications or dosages

Output schema:
{
  "possible_conditions": ["list of conditions"],
  "red_flags": ["dangerous symptom combinations if any, empty list if none"],
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


def create_medical_knowledge_agent_with_rag(llm_config: dict) -> AssistantAgent:
    return AssistantAgent(
        name="MedicalKnowledgeAgent",
        system_message=MEDICAL_KNOWLEDGE_WITH_RAG_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )


def build_rag_prompt(symptoms: dict, rag_context: list[str]) -> str:
    """Combine structured symptom data with WHO reference chunks for the agent."""
    context_block = "\n\n---\n\n".join(rag_context)
    return (
        f"Symptom data:\n{json.dumps(symptoms, indent=2)}\n\n"
        f"Relevant WHO medical reference (use to guide your answer, do not copy verbatim):\n"
        f"{context_block}\n\n"
        "Return JSON output only."
    )
