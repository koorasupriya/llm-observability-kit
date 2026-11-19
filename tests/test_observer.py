import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from llm_obs import LLMObserver, LLMResult
from llm_obs.pricing import estimate_cost

PRICING = {"m1": {"input": 2.0, "output": 10.0}}


@pytest.fixture
def obs():
    spans, metrics = InMemorySpanExporter(), InMemoryMetricReader()
    o = LLMObserver("test", span_exporter=spans, metric_reader=metrics, pricing=PRICING)
    o.spans, o.metrics = spans, metrics
    return o


def metric_points(reader):
    out = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out[m.name] = list(m.data.data_points)
    return out


def test_cost_estimate():
    assert estimate_cost("m1", 1_000_000, 100_000, PRICING) == pytest.approx(3.0)
    assert estimate_cost("unknown", 10, 10, PRICING) is None


def test_llm_span_attributes(obs):
    with obs.llm_call("m1", provider="acme", prompt="hello") as call:
        call.record(LLMResult("hi there", input_tokens=1000, output_tokens=200))
    (span,) = obs.spans.get_finished_spans()
    a = span.attributes
    assert span.name == "chat m1"
    assert a["gen_ai.request.model"] == "m1" and a["gen_ai.system"] == "acme"
    assert a["gen_ai.usage.input_tokens"] == 1000 and a["gen_ai.usage.output_tokens"] == 200
    assert a["llm.token_count.total"] == 1200
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.cost_usd"] == pytest.approx(0.004)
    assert a["input.value"] == "hello" and a["output.value"] == "hi there"


def test_metrics_tokens_cost_latency(obs):
    for _ in range(3):
        with obs.llm_call("m1", provider="acme") as call:
            call.record(LLMResult("x", 100, 50))
    pts = metric_points(obs.metrics)
    tokens = {p.attributes["token.type"]: p.value for p in pts["llm.tokens"]}
    assert tokens == {"input": 300, "output": 150}
    assert sum(p.value for p in pts["llm.cost_usd"]) == pytest.approx(3 * (100 * 2 + 50 * 10) / 1e6)
    assert pts["llm.requests"][0].value == 3
    assert pts["llm.request.duration"][0].count == 3


def test_errors_are_recorded(obs):
    with pytest.raises(TimeoutError):
        with obs.llm_call("m1"):
            raise TimeoutError("slow")
    (span,) = obs.spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == "exception"
    statuses = {p.attributes["status"] for p in metric_points(obs.metrics)["llm.requests"]}
    assert statuses == {"error"}


def test_unpriced_model_flagged(obs):
    with obs.llm_call("mystery") as call:
        call.record(LLMResult("x", 10, 10))
    assert obs.spans.get_finished_spans()[0].attributes["llm.cost_usd.unpriced"] is True


def test_nested_spans_share_trace(obs):
    with obs.span("request"):
        with obs.span("retrieval"):
            pass
        with obs.llm_call("m1") as call:
            call.record(LLMResult("x", 1, 1))
    spans = {s.name: s for s in obs.spans.get_finished_spans()}
    root = spans["request"]
    assert spans["retrieval"].parent.span_id == root.context.span_id
    assert spans["chat m1"].context.trace_id == root.context.trace_id


def test_demo_api_produces_trace(monkeypatch):
    from fastapi.testclient import TestClient

    import llm_obs.demo_app as demo

    spans = InMemorySpanExporter()
    monkeypatch.setattr(demo, "observer", LLMObserver("demo", span_exporter=spans, metric_reader=InMemoryMetricReader()))
    monkeypatch.setattr(demo, "llm", demo.FakeLLM(failure_rate=0.0, seed=1))
    r = TestClient(demo.app).post("/chat", json={"question": "what is rag", "model": "demo-small"})
    assert r.status_code == 200
    names = {s.name for s in spans.get_finished_spans()}
    assert {"POST /chat", "retrieval", "chat demo-small"} <= names
