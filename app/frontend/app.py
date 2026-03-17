import os
import sys

# Ensure the project root is on the path so `app.*` imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from dotenv import load_dotenv
from app.pipeline.triage_pipeline import triage_pipeline_stream
from app.pipeline.tracing import get_tracing_status

load_dotenv()

# ── LLM config ───────────────────────────────────────────────────────────────
LLM_CONFIG = {
    "config_list": [
        {
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ],
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}

RISK_COLORS = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🔴",
    "CRITICAL": "🚨",
}

STEP_ICONS = {
    "pending": "⏳",
    "done": "✅",
    "error": "❌",
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedTriage AI", page_icon="🏥", layout="centered")
st.title("🏥 MedTriage AI")
st.caption("AI-powered health triage assistant — for guidance only, not diagnosis.")

# ── Sidebar: LangSmith tracing status ────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Observability")
    ts = get_tracing_status()
    if ts["connected"]:
        st.success(
            f"**LangSmith active**\n\n"
            f"Project: `{ts['project']}`\n\n"
            f"Endpoint: `{ts['endpoint']}`"
        )
    elif ts["tracing_enabled"] and ts["api_key_set"]:
        st.warning(f"**LangSmith configured** but connection failed:\n\n{ts['error']}")
    else:
        st.info(f"**LangSmith disabled**\n\n{ts['error'] or 'Set LANGSMITH_TRACING=true to enable.'}")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    result = None
    pipeline_error = None

    with st.chat_message("assistant"):
        # ── Live streaming progress panel ─────────────────────────────────────
        with st.status("🔬 Analyzing your symptoms...", expanded=True) as status:
            step_placeholders = {}

            for event in triage_pipeline_stream(user_input, LLM_CONFIG):
                ev = event.get("event")

                if ev == "start":
                    total = event.get("total_steps", 4)
                    st.write(f"Running {total} AI agents in sequence...")

                elif ev == "step_start":
                    step = event["step"]
                    label = event["label"]
                    # Create a placeholder we can update when the step finishes
                    step_placeholders[step] = st.empty()
                    step_placeholders[step].write(f"⏳ **Step {step}:** {label}")

                elif ev == "step_done":
                    step = event["step"]
                    label = event["label"]
                    if step in step_placeholders:
                        step_placeholders[step].write(f"✅ **Step {step}:** {label}")

                elif ev == "done":
                    result = event["data"]
                    status.update(
                        label="✅ Analysis complete!",
                        state="complete",
                        expanded=False,
                    )

                elif ev == "error":
                    pipeline_error = event.get("detail", "Unknown error")
                    status.update(label="❌ Analysis failed", state="error")
                    break

        # ── Error display ─────────────────────────────────────────────────────
        if pipeline_error:
            st.error(f"Pipeline error: {pipeline_error}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {pipeline_error}"}
            )

        # ── Result display ────────────────────────────────────────────────────
        elif result:
            risk = result["risk"]
            response = result["response"]
            structured = result["structured_symptoms"]
            knowledge = result["knowledge"]

            risk_level = risk.get("risk_level", "UNKNOWN")
            badge = RISK_COLORS.get(risk_level, "⚪")

            st.subheader(f"Risk Level: {badge} {risk_level}")
            st.write(f"**Reason:** {risk.get('reason', '')}")
            st.markdown("---")
            st.write(f"**Advice:** {response.get('advice', '')}")

            action_steps = response.get("action_steps") or []
            if action_steps:
                st.markdown("**Action steps:**")
                for step in action_steps:
                    st.write(f"• {step}")

            st.warning(response.get("disclaimer", ""))

            with st.expander("🔍 Extracted Symptom Data"):
                st.json(structured)

            with st.expander("📚 Possible Conditions & Red Flags"):
                st.json(knowledge)

            # Store compact summary in chat history
            summary = (
                f"**Risk Level: {badge} {risk_level}**\n\n"
                f"**Reason:** {risk.get('reason', '')}\n\n"
                f"**Advice:** {response.get('advice', '')}"
            )
            st.session_state.messages.append({"role": "assistant", "content": summary})
