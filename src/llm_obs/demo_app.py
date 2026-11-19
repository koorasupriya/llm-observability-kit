"""Demo FastAPI service: every request produces HTTP -> retrieval -> LLM spans and LLM metrics.

    uvicorn llm_obs.demo_app:app
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uvicorn llm_obs.demo_app:app   # send to the stack
"""
from __future__ import annotations

import asyncio
import random
import time

from fastapi import FastAPI, HTTPException, Request
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel

from .observer import LLMObserver, LLMResult

observer = LLMObserver(service_name="llm-demo-api")
app = FastAPI(title="LLM Observability Demo")


class FakeLLM:
    """Stands in for a real provider so the demo needs no API key: simulated latency, tokens and failures."""

    def __init__(self, failure_rate: float = 0.05, seed: int | None = None):
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)

    async def generate(self, model: str, prompt: str) -> LLMResult:
        await asyncio.sleep(self.rng.uniform(0.05, 0.4) * (3 if model == "demo-large" else 1))
        if self.rng.random() < self.failure_rate:
            raise TimeoutError("upstream LLM timeout")
        out = f"[{model}] answer to: {prompt[:60]}"
        return LLMResult(out, input_tokens=len(prompt.split()) * 4 + 120,
                         output_tokens=self.rng.randint(40, 400))


llm = FakeLLM()


@app.middleware("http")
async def trace_http(request: Request, call_next):
    with observer.tracer.start_as_current_span(
        f"{request.method} {request.url.path}", kind=SpanKind.SERVER,
        attributes={"http.request.method": request.method, "url.path": request.url.path},
    ) as span:
        start = time.perf_counter()
        response = await call_next(request)
        span.set_attribute("http.response.status_code", response.status_code)
        span.set_attribute("http.server.duration_ms", round((time.perf_counter() - start) * 1000, 2))
        if response.status_code >= 500:
            span.set_status(Status(StatusCode.ERROR))
        return response


class ChatRequest(BaseModel):
    question: str
    model: str = "demo-small"


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    with observer.span("retrieval", **{"retrieval.top_k": 4}) as s:
        await asyncio.sleep(0.02)
        s.set_attribute("retrieval.documents", 4)
    try:
        with observer.llm_call(req.model, provider="demo", prompt=req.question) as call:
            result = await llm.generate(req.model, req.question)
            call.record(result)
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    return {"answer": result.text, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
