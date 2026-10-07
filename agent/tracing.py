"""Arize Phoenix: OpenTelemetry + OpenInference auto-instrumentation для LangChain/LangGraph."""
import os

from agent import config

_tracer = None


def setup_tracing():
    """Повертає OTel tracer. Без PHOENIX_COLLECTOR_ENDPOINT (наприклад, у CI) — no-op tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer
    from opentelemetry import trace
    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("DISABLE_TRACING") == "1":
        _tracer = trace.get_tracer("retention-agent")
        return _tracer
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    tp = register(project_name=config.PHOENIX_PROJECT_NAME, batch=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=tp)
    _tracer = tp.get_tracer("retention-agent")
    return _tracer
