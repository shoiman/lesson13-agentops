"""SLA middleware (Lab 3 + Part 2): токен-бюджет з hard-stop, soft warning, circuit breaker."""
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace

from agent import config
from agent.alerts import send_alert


class BudgetExceeded(Exception):
    pass


class BudgetMiddleware(BaseCallbackHandler):
    """LangChain callback: рахує $ після кожного LLM-виклику і зупиняє run при перевищенні бюджету.

    raise_error=True — виняток з callback-а не ковтається, а перериває граф LangGraph (hard-stop).
    Бюджет змінний: стартує з DEFAULT_RUN_BUDGET_USD, а tool get_customer_churn_risk
    перевстановлює його за churn-tier (high $0.50 / medium $0.15 / low $0.05).
    """
    raise_error = True

    def __init__(self, budget_usd: float = config.DEFAULT_RUN_BUDGET_USD, run_id: str = "",
                 fixed: bool = False):
        self.budget_usd = budget_usd
        self.fixed = fixed            # True → бюджет задано вручну (--budget), tier його не змінює
        self.run_id = run_id
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.soft_warned = False

    def set_budget(self, budget_usd: float, reason: str) -> None:
        if self.fixed:
            return
        self.budget_usd = budget_usd
        trace.get_current_span().add_event("budget.updated", {"budget_usd": budget_usd, "reason": reason})

    def _check(self) -> None:
        if self.cost_usd > self.budget_usd:
            send_alert("Budget hard-stop", {"run_id": self.run_id, "cost_usd": round(self.cost_usd, 6),
                                            "budget_usd": self.budget_usd})
            raise BudgetExceeded(f"cost ${self.cost_usd:.6f} > budget ${self.budget_usd:.4f}")
        if not self.soft_warned and self.cost_usd > config.SOFT_BUDGET_RATIO * self.budget_usd:
            self.soft_warned = True
            print(f"[SLA] soft budget warning: ${self.cost_usd:.6f} > 80% of ${self.budget_usd}")

    def on_chat_model_start(self, serialized: Any, messages: Any, **kw: Any) -> None:
        self._check()   # не починати новий LLM-виклик, якщо бюджет вже вичерпано

    def on_llm_end(self, response: Any, **kw: Any) -> None:
        self.llm_calls += 1
        inp = out = 0
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                inp += um.get("input_tokens", 0)
                out += um.get("output_tokens", 0)
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += (inp * config.PRICE_INPUT_PER_1M + out * config.PRICE_OUTPUT_PER_1M) / 1e6
        self._check()


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    """CLOSED → (N поспіль помилок) → OPEN → (timeout) → HALF_OPEN → успіх: CLOSED / помилка: OPEN."""

    def __init__(self, name: str, threshold: int = config.CB_FAILURE_THRESHOLD,
                 reset_timeout_s: float = config.CB_RESET_TIMEOUT_S):
        self.name, self.threshold, self.reset_timeout_s = name, threshold, reset_timeout_s
        self.state, self.failures, self.opened_at = "CLOSED", 0, 0.0

    def _transition(self, new_state: str, reason: str) -> None:
        old, self.state = self.state, new_state
        print(f"[CIRCUIT] {self.name}: {old} -> {new_state} ({reason})")
        trace.get_current_span().add_event("circuit_breaker.state_change",
                                           {"tool": self.name, "from": old, "to": new_state})
        if new_state == "OPEN":
            send_alert(f"Circuit breaker OPEN: {self.name}",
                       {"tool": self.name, "from": old, "consecutive_failures": self.failures,
                        "reason": reason})

    def call(self, fn, *args, **kwargs):
        if self.state == "OPEN":
            if time.monotonic() - self.opened_at >= self.reset_timeout_s:
                self._transition("HALF_OPEN", "reset timeout elapsed, trial request")
            else:
                raise CircuitOpenError(f"circuit for '{self.name}' is OPEN — tool temporarily disabled")
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            self.failures += 1
            if self.state == "HALF_OPEN" or self.failures >= self.threshold:
                self.opened_at = time.monotonic()
                self._transition("OPEN", f"{type(e).__name__}: {e}")
            raise
        if self.state != "CLOSED":
            self._transition("CLOSED", "trial request succeeded")
        self.failures = 0
        return result


BREAKERS: dict[str, CircuitBreaker] = {}


def breaker(name: str) -> CircuitBreaker:
    return BREAKERS.setdefault(name, CircuitBreaker(name))
