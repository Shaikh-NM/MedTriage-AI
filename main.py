import os
from dotenv import load_dotenv
from app.pipeline.triage_pipeline import triage_pipeline

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ],
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    result = triage_pipeline(user_input, llm_config)

    print("\n=== RISK LEVEL:", result["risk"]["risk_level"], "===")
    print("Reason:", result["risk"]["reason"])
    print("\nAdvice:", result["response"]["advice"])
    for step in result["response"].get("action_steps") or []:
        print(f"  • {step}")
    print("\nDisclaimer:", result["response"]["disclaimer"])
