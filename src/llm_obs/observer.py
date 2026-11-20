"""Core instrumentation: one object that owns a tracer and a meter for an LLM service.

Every LLM call becomes a span carrying OpenTelemetry GenAI semantic-convention attributes
(`gen_ai.*`) plus OpenInference attributes so Arize Phoenix renders it as an LLM span.
The same call also updates metrics (tokens, cost, latency, errors) for Grafana dashboards.
"""
from __future__ import annotations

import functools
import inspect
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from opentelemetry import trace
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

from .pricing import estimate_cost, load_pricing


@dataclass
class LLMResult:
    """What an instrumented call must report back."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str | None = None  # the model that actually answered, if it differs from the request


class LLMCallRecorder:
    """Handed to the caller inside `observer.llm_call(...)`; call `.record(result)` once."""

    def __init__(self, span: trace.Span):
        self.span = span
        self.result: LLMResult | None = None

    def record(self, result: LLMResult) -> None:
        self.result = result


def _otlp_span_exporter(endpoint: str) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")


def _otlp_metric_reader(endpoint: str) -> MetricReader:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    return PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint.rstrip('/')}/v1/metrics"), export_interval_millis=10_000
    )


class LLMObserver:
    def __init__(
        self,
        service_name: str = "llm-service",
        span_exporter: SpanExporter | None = None,
        metric_reader: MetricReader | None = None,
        otlp_endpoint: str | None = None,
        pricing: dict | None = None,
    ):
        resource = Resource.create({"service.name": service_name})
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

        self.tracer_provider = TracerProvider(resource=resource)
        if span_exporter is not None:  # tests / console: export immediately
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        elif otlp_endpoint:
            self.tracer_provider.add_span_processor(BatchSpanProcessor(_otlp_span_exporter(otlp_endpoint)))

        readers = [metric_reader] if metric_reader else ([_otlp_metric_reader(otlp_endpoint)] if otlp_endpoint else [])
        self.meter_provider = MeterProvider(resource=resource, metric_readers=readers)

        self.tracer = self.tracer_provider.get_tracer("llm_obs")
        meter: Meter = self.meter_provider.get_meter("llm_obs")
        self.pricing = pricing if pricing is not None else load_pricing()

        self.tokens = meter.create_counter("llm.tokens", unit="{token}", description="Tokens consumed")
        self.cost = meter.create_counter("llm.cost_usd", unit="1", description="Estimated cost in USD")
        self.requests = meter.create_counter("llm.requests", unit="{request}", description="LLM calls")
        self.latency = meter.create_histogram("llm.request.duration", unit="s", description="LLM call latency")

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[trace.Span]:
        """A generic child span (retrieval, tool call, post-processing...)."""
        with self.tracer.start_as_current_span(name, attributes=attributes) as s:
            yield s

    def traced(self, name: str | None = None, kind: str = "CHAIN", **attributes):
        """Decorator that wraps a sync or async function in a span.

            @observer.traced(kind="RETRIEVER")
            async def search(query): ...

        `kind` follows OpenInference span kinds (CHAIN, RETRIEVER, TOOL, EMBEDDING, ...), so Phoenix
        shows the step with the right icon. Exceptions are recorded on the span and re-raised.
        """
        def decorate(fn):
            span_name = name or fn.__qualname__
            attrs = {"openinference.span.kind": kind, "code.function": fn.__qualname__, **attributes}

            def start():
                return self.tracer.start_as_current_span(span_name, attributes=attrs,
                                                         record_exception=True, set_status_on_exception=True)

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with start():
                        return await fn(*args, **kwargs)
                return async_wrapper

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with start():
                    return fn(*args, **kwargs)
            return wrapper

        return decorate

    @contextmanager
    def llm_call(self, model: str, provider: str = "unknown", prompt: str | None = None,
                 operation: str = "chat") -> Iterator[LLMCallRecorder]:
        """Wrap a single LLM request:

            with observer.llm_call("demo-small", provider="demo", prompt=q) as call:
                resp = client.generate(q)
                call.record(LLMResult(resp.text, resp.in_tokens, resp.out_tokens))
        """
        attrs = {
            "gen_ai.operation.name": operation,
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "openinference.span.kind": "LLM",
            "llm.model_name": model,
        }
        if prompt is not None:
            attrs["input.value"] = prompt[:4000]
        start = time.perf_counter()
        status = "ok"
        with self.tracer.start_as_current_span(f"{operation} {model}", attributes=attrs) as span:
            recorder = LLMCallRecorder(span)
            try:
                yield recorder
            except Exception as exc:
                status = "error"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                elapsed = time.perf_counter() - start
                labels = {"gen_ai.request.model": model, "gen_ai.system": provider, "status": status}
                self.requests.add(1, labels)
                self.latency.record(elapsed, labels)
                if recorder.result is not None:
                    self._finish(span, recorder.result, model, provider)

    def _finish(self, span: trace.Span, r: LLMResult, model: str, provider: str) -> None:
        used_model = r.model or model
        cost = estimate_cost(used_model, r.input_tokens, r.output_tokens, self.pricing)
        span.set_attributes({
            "gen_ai.response.model": used_model,
            "gen_ai.usage.input_tokens": r.input_tokens,
            "gen_ai.usage.output_tokens": r.output_tokens,
            "llm.token_count.prompt": r.input_tokens,
            "llm.token_count.completion": r.output_tokens,
            "llm.token_count.total": r.input_tokens + r.output_tokens,
            "output.value": r.text[:4000],
        })
        labels = {"gen_ai.request.model": used_model, "gen_ai.system": provider}
        self.tokens.add(r.input_tokens, {**labels, "token.type": "input"})
        self.tokens.add(r.output_tokens, {**labels, "token.type": "output"})
        if cost is not None:
            span.set_attribute("llm.cost_usd", cost)
            self.cost.add(cost, labels)
        else:
            span.set_attribute("llm.cost_usd.unpriced", True)

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()
