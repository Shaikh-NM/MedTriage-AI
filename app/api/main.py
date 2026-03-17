import os
import json
import queue
import asyncio
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.models.symptom_input import SymptomInput
from app.pipeline.triage_pipeline import triage_pipeline, triage_pipeline_stream
from app.pipeline.tracing import get_tracing_status, print_tracing_status

load_dotenv()
print_tracing_status()  # log tracing status once at startup

app = FastAPI(
    title="MedTriage AI API",
    description="AI-powered medical triage — for guidance only, not diagnosis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LLM_CONFIG = {
    "config_list": [
        {
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ],
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}


class TriageRequest(BaseModel):
    """Accepts either a structured SymptomInput or a free-text query."""
    free_text: Optional[str] = None
    structured: Optional[SymptomInput] = None


class TriageResponse(BaseModel):
    structured_symptoms: dict
    knowledge: dict
    risk: dict
    response: dict


def _resolve_input(request: TriageRequest) -> str:
    if not request.free_text and not request.structured:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'free_text' or 'structured' input.",
        )
    return request.structured.to_prompt_string() if request.structured else request.free_text


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "gpt-4o")}


@app.get("/langsmith/status")
async def langsmith_status():
    """Return current LangSmith tracing configuration and connectivity."""
    return get_tracing_status()


# ── Blocking endpoint (returns full result at once) ───────────────────────────

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """Run the full pipeline and return the complete result when done."""
    user_input = _resolve_input(request)
    try:
        result = await asyncio.to_thread(triage_pipeline, user_input, _LLM_CONFIG)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/triage/structured", response_model=TriageResponse)
async def triage_structured(data: SymptomInput):
    """Convenience endpoint — accepts a SymptomInput directly."""
    try:
        result = await asyncio.to_thread(triage_pipeline, data.to_prompt_string(), _LLM_CONFIG)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming endpoint (Server-Sent Events) ───────────────────────────────────

@app.post("/triage/stream")
async def triage_stream(request: TriageRequest):
    """
    Stream pipeline progress as Server-Sent Events (SSE).

    Each event is a JSON-encoded dict on a `data:` line:
      data: {"event": "start",      "total_steps": 4}
      data: {"event": "step_start", "step": 1, "label": "..."}
      data: {"event": "step_done",  "step": 1, "label": "...", "data": {...}}
      ...
      data: {"event": "done",       "data": {full result}}
      data: {"event": "error",      "detail": "error message"}

    Each message is separated by a blank line (standard SSE format).
    """
    user_input = _resolve_input(request)
    result_queue: queue.Queue = queue.Queue()

    def _run_in_thread():
        """Run the synchronous generator in a background thread, pushing to queue."""
        try:
            for event in triage_pipeline_stream(user_input, _LLM_CONFIG):
                result_queue.put(event)
        except Exception as e:
            result_queue.put({"event": "error", "detail": str(e)})
        finally:
            result_queue.put(None)  # sentinel: stream is finished

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    async def event_generator():
        while True:
            # Read from the queue without blocking the async event loop
            event = await asyncio.to_thread(result_queue.get)
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        thread.join()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering if behind a proxy
        },
    )
