"""Lab 3 демо: chaos — ML-сервіс "падає", circuit breaker переходить у OPEN → Slack-алерт.

Без LLM (детерміновано, безкоштовно). Спани йдуть у Phoenix.
  python scripts/demo_circuit_breaker.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
from agent.sla import BudgetMiddleware, breaker  # noqa: E402
from agent.tools import RunContext, build_tools  # noqa: E402
from agent.tracing import setup_tracing  # noqa: E402

tracer = setup_tracing()
cb = breaker("get_customer_churn_risk")
cb.reset_timeout_s = 3
risk_tool = build_tools(RunContext(budget=BudgetMiddleware()))[0]

with tracer.start_as_current_span("circuit_breaker_chaos_demo") as span:
    span.set_attribute("openinference.span.kind", "CHAIN")
    os.environ["CHAOS_ML_TOOL_FAIL"] = "1"
    for i in range(1, 6):
        out = risk_tool.invoke({"customer_id": "4097-BGZQO"})
        print(f"call {i}: state={cb.state:<9} -> {out[:90]}")
    print(f"\n...ML service recovered, waiting {cb.reset_timeout_s}s for HALF_OPEN trial...")
    os.environ["CHAOS_ML_TOOL_FAIL"] = "0"
    time.sleep(cb.reset_timeout_s)
    out = risk_tool.invoke({"customer_id": "4097-BGZQO"})
    print(f"call 6: state={cb.state:<9} -> {out[:90]}")
time.sleep(2)
