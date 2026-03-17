import json
from agents.symptom_intake_agent import create_symptom_intake_agent
from agents.medical_knowledge_agent import create_medical_knowledge_agent
from agents.risk_assessment_agent import create_risk_assessment_agent
from agents.escalation_agent import create_escalation_agent
from autogen import UserProxyAgent

def run_agent(agent, user_proxy, message: str) -> dict:
    user_proxy.initiate_chat(agent, message=message, max_turns=1)
    last_msg = user_proxy.last_message(agent)["content"]
    return json.loads(last_msg)

def triage_pipeline(user_input: str, llm_config: dict) -> dict:
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Step 1: Symptom Intake
    symptom_agent = create_symptom_intake_agent(llm_config)
    structured = run_agent(symptom_agent, user_proxy, user_input)

    # Step 2: Medical Knowledge
    knowledge_agent = create_medical_knowledge_agent(llm_config)
    knowledge = run_agent(knowledge_agent, user_proxy, json.dumps(structured))

    # Step 3: Risk Assessment
    risk_agent = create_risk_assessment_agent(llm_config)
    combined_input = json.dumps({**structured, **knowledge})
    risk = run_agent(risk_agent, user_proxy, combined_input)

    # Step 4: Escalation
    escalation_agent = create_escalation_agent(llm_config)
    response = run_agent(escalation_agent, user_proxy, json.dumps(risk))

    return {
        "structured_symptoms": structured,
        "knowledge": knowledge,
        "risk": risk,
        "response": response,
    }