"""Tools агента. Створюються на кожен run (closure над RunContext), обгорнуті circuit breaker-ом."""
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from langchain_core.tools import tool

from agent import config
from agent.alerts import send_alert
from agent.guards import GuardrailViolation, check_domain
from agent.policy import evaluate_offer
from agent.sla import BudgetMiddleware, CircuitOpenError, breaker

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def _model():
    return joblib.load(ROOT / "models" / "churn_model.joblib")


@lru_cache
def _crm() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "customers_crm.csv").set_index("customerID")


def known_customer_names() -> list:
    return _crm()["full_name"].dropna().unique().tolist()


@dataclass
class RunContext:
    budget: BudgetMiddleware
    customer_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_tier: Optional[str] = None
    offer: Optional[dict] = None
    tool_calls: list = field(default_factory=list)


def _protected(name: str, ctx: RunContext, fn, *args):
    """Circuit breaker + журнал викликів. Помилку повертаємо LLM текстом, а не валимо граф."""
    ctx.tool_calls.append(name)
    try:
        return breaker(name).call(fn, *args)
    except CircuitOpenError as e:
        return f"ERROR: {e}. Do not retry this tool; use a fallback or tell the customer a manager will follow up."
    except GuardrailViolation as e:
        return f"BLOCKED BY GUARDRAIL: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def build_tools(ctx: RunContext):

    def _risk(customer_id: str) -> str:
        if os.getenv("CHAOS_ML_TOOL_FAIL") == "1":          # chaos engineering: ML-сервіс "впав"
            raise ConnectionError("churn model service unavailable (chaos)")
        if customer_id not in _crm().index:
            raise KeyError(f"customer '{customer_id}' not found")
        m = _model()
        row = _crm().loc[[customer_id], m["features"]]
        score = float(m["pipeline"].predict_proba(row)[0, 1])
        tier, budget = config.tier_for(score)
        ctx.customer_id, ctx.risk_score, ctx.risk_tier = customer_id, score, tier
        ctx.budget.set_budget(budget, f"risk tier {tier}")
        r = row.iloc[0]
        return json.dumps({
            "customer_id": customer_id, "churn_risk_score": round(score, 3), "risk_tier": tier,
            "reasoning_budget_usd": ctx.budget.budget_usd,
            "max_discount_allowed_pct": (config.MAX_DISCOUNT_ABSOLUTE_PCT
                                         if score >= config.BIG_DISCOUNT_RISK_THRESHOLD
                                         else config.MAX_DISCOUNT_LOW_RISK_PCT),
            "key_factors": {"contract": r["Contract"], "tenure_months": int(r["tenure"]),
                            "monthly_charges": float(r["MonthlyCharges"]),
                            "internet": r["InternetService"], "tech_support": r["TechSupport"]},
        })

    def _profile(customer_id: str) -> str:
        if customer_id not in _crm().index:
            raise KeyError(f"customer '{customer_id}' not found")
        return _crm().loc[customer_id].to_json()   # містить PII: full_name, email, phone

    def _search(url: str, query: str) -> str:
        check_domain(url)                            # guardrail: allowlist доменів
        docs = json.loads((ROOT / "data" / "retention_playbook.json").read_text(encoding="utf-8"))
        tier = (ctx.risk_tier or "any")
        hits = [d for d in docs if d["tier"] in (tier, "any")] or docs
        return json.dumps(hits, ensure_ascii=False)

    def _offer(customer_id: str, discount_pct: float, free_months: int, bonus_value_usd: float,
               rationale: str) -> str:
        if ctx.risk_score is None or customer_id != ctx.customer_id:
            raise ValueError("call get_customer_churn_risk for this customer first")
        monthly = float(_crm().loc[customer_id, "MonthlyCharges"])
        d = evaluate_offer(ctx.risk_score, monthly, discount_pct, free_months, bonus_value_usd)
        ctx.offer = {**d.to_dict(), "customer_id": customer_id, "rationale": rationale,
                     "risk_score": round(ctx.risk_score, 3)}
        if d.status == "PENDING_HUMAN_REVIEW":
            with open(ROOT / "escalations.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ctx.offer, ensure_ascii=False) + "\n")
            send_alert("Human review required (retention offer)", ctx.offer)
        return json.dumps(ctx.offer, ensure_ascii=False)

    @tool
    def get_customer_churn_risk(customer_id: str) -> str:
        """Predict churn risk for a customer with the pre-trained ML model.
        ALWAYS call this first. Returns churn_risk_score (0..1), risk_tier (high/medium/low),
        max_discount_allowed_pct and key churn factors (contract, tenure, charges).
        customer_id example: '2089-ZDWBI'."""
        return _protected("get_customer_churn_risk", ctx, _risk, customer_id)

    @tool
    def get_customer_profile(customer_id: str) -> str:
        """Fetch the CRM profile of a customer: subscribed services, contract, payment method,
        plus contact data (name, email, phone). Contact data is CONFIDENTIAL — never repeat it."""
        return _protected("get_customer_profile", ctx, _profile, customer_id)

    @tool
    def search_retention_playbook(url: str, query: str) -> str:
        """Search the internal retention playbook for strategies suitable for the customer's tier.
        Only internal https domains are allowed, use url='https://kb.telco-retention.internal/playbook'."""
        return _protected("search_retention_playbook", ctx, _search, url, query)

    @tool
    def propose_retention_offer(customer_id: str, discount_pct: float, free_months: int,
                                bonus_value_usd: float, rationale: str) -> str:
        """Submit ONE retention offer for policy validation. discount_pct: monthly discount in %,
        free_months: number of free service months, bonus_value_usd: value of extra bonuses in USD.
        Returns status APPROVED, REJECTED (policy violation — you may submit a corrected offer)
        or PENDING_HUMAN_REVIEW (a manager must approve it)."""
        return _protected("propose_retention_offer", ctx, _offer, customer_id, discount_pct,
                          free_months, bonus_value_usd, rationale)

    return [get_customer_churn_risk, get_customer_profile, search_retention_playbook,
            propose_retention_offer]
