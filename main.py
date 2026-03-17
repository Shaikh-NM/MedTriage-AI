import os
from dotenv import load_dotenv
from pipeline.triage_pipeline import triage_pipeline

load_dotenv()

llm_config = {
    "config_list": [
        {"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}
    ],
    "temperature": 0.1,
}

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    result = triage_pipeline(user_input, llm_config)
    print("\n=== RISK LEVEL:", result["risk"]["risk_level"], "===")
    print("Advice:", result["response"]["advice"])
    print("\nDisclaimer:", result["response"]["disclaimer"])