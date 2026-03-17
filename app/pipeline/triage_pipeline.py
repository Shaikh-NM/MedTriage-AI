import json
from typing import Generator
from autogen import UserProxyAgent
from langsmith import traceable
from app.agents.symptom_intake_agent import create_symptom_intake_agent
from app.agents.medical_knowledge_agent import (
    create_medical_knowledge_agent_with_rag,
    build_rag_prompt,
)
from app.agents.risk_assessment_agent import create_risk_assessment_agent
from app.agents.escalation_agent import create_escalation_agent
from app.pipeline.guardrails import enforce_guardrails, escalation_bias
from app.rag.retriever import MedicalRetriever

# Lazy singleton — loaded once on first pipeline call
_retriever: MedicalRetriever | None = None


def _get_retriever() -> MedicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever


def _run_agent(agent, user_proxy: UserProxyAgent, message: str) -> dict:
    """Send one message to an agent and parse its JSON reply."""
    user_proxy.initiate_chat(agent, message=message, max_turns=1)
    last_msg = user_proxy.last_message(agent)["content"]

    # Strip markdown code fences if the model wraps JSON in them
    content = last_msg.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)


# ── Per-step traceable wrappers ───────────────────────────────────────────────
# Each wrapper creates a named child span in LangSmith, making it easy to see
# exactly which agent ran, what it received, and what it returned.

@traceable(run_type="chain", name="1 · SymptomIntake")
def _step_symptom_intake(agent, user_proxy: UserProxyAgent, user_input: str) -> dict:
    return _run_agent(agent, user_proxy, user_input)


@traceable(run_type="chain", name="2 · MedicalKnowledge+RAG")
def _step_medical_knowledge(agent, user_proxy: UserProxyAgent, prompt: str) -> dict:
    return _run_agent(agent, user_proxy, prompt)


@traceable(run_type="chain", name="3 · RiskAssessment")
def _step_risk_assessment(agent, user_proxy: UserProxyAgent, combined: str) -> dict:
    return _run_agent(agent, user_proxy, combined)


@traceable(run_type="chain", name="4 · EscalationAdvice")
def _step_escalation(agent, user_proxy: UserProxyAgent, risk_json: str) -> dict:
    return _run_agent(agent, user_proxy, risk_json)


def _apply_demographic_adjustments(structured: dict, risk: dict) -> dict:
    """Escalate risk for vulnerable demographics — infants, elderly, high-risk conditions."""
    age = structured.get("age")
    conditions = [c.lower() for c in (structured.get("existing_conditions") or [])]

    vulnerable_conditions = {"pregnancy", "pregnant", "diabetes", "heart disease", "immunocompromised", "cancer"}

    if age is not None and (age < 5 or age > 65):
        if risk["risk_level"] == "LOW":
            risk["risk_level"] = "MEDIUM"
            risk["reason"] += " (escalated: vulnerable age group)"

    if vulnerable_conditions & set(conditions):
        if risk["risk_level"] in ("LOW", "MEDIUM"):
            risk["risk_level"] = "HIGH"
            risk["reason"] += " (escalated: high-risk existing condition)"

    return risk


def triage_pipeline_stream(
    user_input: str, llm_config: dict
) -> Generator[dict, None, None]:
    """
    Streaming version of the triage pipeline.

    Yields event dicts so callers can show live progress:
      {"event": "start",      "total_steps": 4}
      {"event": "step_start", "step": N, "label": "..."}
      {"event": "step_done",  "step": N, "label": "...", "data": {...}}
      {"event": "done",       "data": {full result dict}}
      {"event": "error",      "detail": "error message"}
    """
    try:
        yield {"event": "start", "total_steps": 4}

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        # ── Step 1: Symptom extraction ────────────────────────────────────────
        yield {"event": "step_start", "step": 1, "label": "Extracting symptoms from your description..."}
        symptom_agent = create_symptom_intake_agent(llm_config)
        structured = _step_symptom_intake(symptom_agent, user_proxy, user_input)
        yield {
            "event": "step_done",
            "step": 1,
            "label": f"Symptoms extracted: {', '.join(structured.get('symptoms', []))}",
            "data": structured,
        }

        # ── Step 2: Medical knowledge + RAG ──────────────────────────────────
        yield {"event": "step_start", "step": 2, "label": "Searching WHO guidelines and identifying conditions..."}
        retriever = _get_retriever()
        symptom_text = ", ".join(structured.get("symptoms") or [user_input])
        rag_docs = retriever.retrieve(symptom_text, k=5)
        knowledge_prompt = build_rag_prompt(structured, rag_docs)
        knowledge_agent = create_medical_knowledge_agent_with_rag(llm_config)
        knowledge = _step_medical_knowledge(knowledge_agent, user_proxy, knowledge_prompt)
        conditions = ", ".join(knowledge.get("possible_conditions", []))
        yield {
            "event": "step_done",
            "step": 2,
            "label": f"Possible conditions identified: {conditions}",
            "data": knowledge,
        }

        # ── Step 3: Risk assessment ───────────────────────────────────────────
        yield {"event": "step_start", "step": 3, "label": "Assessing risk level..."}
        risk_agent = create_risk_assessment_agent(llm_config)
        combined = json.dumps({**structured, **knowledge})
        risk = _step_risk_assessment(risk_agent, user_proxy, combined)

        # Post-processing before generating advice
        risk = _apply_demographic_adjustments(structured, risk)
        if risk.get("uncertainty"):
            risk["risk_level"] = escalation_bias(risk["risk_level"], uncertainty_flag=True)

        yield {
            "event": "step_done",
            "step": 3,
            "label": f"Risk level determined: {risk.get('risk_level', 'UNKNOWN')}",
            "data": risk,
        }

        # ── Step 4: Escalation advice ─────────────────────────────────────────
        yield {"event": "step_start", "step": 4, "label": "Generating personalised advice..."}
        escalation_agent = create_escalation_agent(llm_config)
        response = _step_escalation(escalation_agent, user_proxy, json.dumps(risk))
        response = enforce_guardrails(response)
        yield {
            "event": "step_done",
            "step": 4,
            "label": "Advice ready",
            "data": response,
        }

        # ── Final result ──────────────────────────────────────────────────────
        yield {
            "event": "done",
            "data": {
                "structured_symptoms": structured,
                "knowledge": knowledge,
                "risk": risk,
                "response": response,
            },
        }

    except Exception as e:
        yield {"event": "error", "detail": str(e)}


@traceable(run_type="chain", name="MedTriage · Full Pipeline")
def triage_pipeline(user_input: str, llm_config: dict) -> dict:
    """
    Blocking wrapper around triage_pipeline_stream.
    Returns the final result dict once all steps complete.
    Used by main.py and tests.
    """
    result = None
    for event in triage_pipeline_stream(user_input, llm_config):
        if event["event"] == "done":
            result = event["data"]
        elif event["event"] == "error":
            raise RuntimeError(event["detail"])
    return result
