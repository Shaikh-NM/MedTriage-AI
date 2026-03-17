"""
LangSmith tracing configuration and connectivity verification.

LangSmith traces every agent step in the triage pipeline.
Set these variables in your .env to enable tracing:

    LANGCHAIN_API_KEY=lsv2_...          # your LangSmith API key
    LANGSMITH_TRACING=true              # enable langsmith SDK tracing
    LANGCHAIN_TRACING_V2=true           # enable LangChain component tracing
    LANGCHAIN_PROJECT=medtriageai       # project name in LangSmith
    LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # EU endpoint
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_tracing_status() -> dict:
    """
    Return a dict describing the current LangSmith tracing configuration
    and whether a live connection can be established.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        or os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )

    status = {
        "tracing_enabled": tracing_enabled,
        "api_key_set": bool(api_key),
        "endpoint": endpoint,
        "project": project,
        "connected": False,
        "error": None,
    }

    if not api_key:
        status["error"] = "LANGCHAIN_API_KEY is not set"
        return status

    if not tracing_enabled:
        status["error"] = "Neither LANGSMITH_TRACING nor LANGCHAIN_TRACING_V2 is set to 'true'"
        return status

    try:
        from langsmith import Client
        client = Client(api_url=endpoint, api_key=api_key)
        # Lightweight check — list at most 1 project
        list(client.list_projects(limit=1))
        status["connected"] = True
    except Exception as e:
        status["error"] = str(e)

    return status


def print_tracing_status() -> None:
    """Print a human-readable tracing status summary (used at startup)."""
    s = get_tracing_status()
    if s["connected"]:
        print(
            f"[LangSmith] OK - Tracing active | project='{s['project']}' "
            f"endpoint={s['endpoint']}"
        )
    elif s["tracing_enabled"] and s["api_key_set"]:
        print(f"[LangSmith] WARN - Tracing configured but connection failed: {s['error']}")
    else:
        print(f"[LangSmith] INFO - Tracing disabled. {s['error'] or ''}")
