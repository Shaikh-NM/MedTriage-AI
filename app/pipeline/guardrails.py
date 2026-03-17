FORBIDDEN_PATTERNS = [
    "take X mg", "prescribe", "dosage", "you have", "you are diagnosed",
    "definitely", "certainly has", "cure", "treatment is",
]

MANDATORY_DISCLAIMER = (
    "⚠️ This is not a medical diagnosis. This system provides general health "
    "guidance only. Please consult a qualified healthcare professional for "
    "proper evaluation and treatment."
)

def enforce_guardrails(response: dict) -> dict:
    advice = response.get("advice", "")
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in advice.lower():
            response["advice"] = "[Response filtered for safety] " + advice
            response["guardrail_triggered"] = True
            break
    response["disclaimer"] = MANDATORY_DISCLAIMER
    return response

def escalation_bias(risk_level: str, uncertainty_flag: bool) -> str:
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if uncertainty_flag and risk_level in levels[:-1]:
        return levels[levels.index(risk_level) + 1]
    return risk_level