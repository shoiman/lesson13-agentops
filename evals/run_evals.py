"""Eval regression suite (Lab 4) + Threshold Gate.

Кожен кейс golden dataset проганяється через поточну версію агента, далі — детерміновані
перевірки-інваріанти (rule-based judge). Gate: accuracy >= ACCURACY_THRESHOLD і
relevance >= RELEVANCE_THRESHOLD, інакше exit code 1 → GitHub Actions блокує PR.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
from agent.guards import EMAIL_RE, PHONE_RE  # noqa: E402
from agent.runner import run_retention  # noqa: E402
from agent.tools import _crm  # noqa: E402

ACCURACY_THRESHOLD = float(os.getenv("EVAL_ACCURACY_THRESHOLD", "0.85"))
RELEVANCE_THRESHOLD = float(os.getenv("EVAL_RELEVANCE_THRESHOLD", "0.90"))
SLEEP_S = float(os.getenv("EVAL_SLEEP_S", "5"))
MAX_ATTEMPTS = int(os.getenv("EVAL_MAX_ATTEMPTS", "4"))
RETRY_SLEEP_S = float(os.getenv("EVAL_RETRY_SLEEP_S", "20"))


def checks_for(case: dict, r: dict) -> dict:
    offer = r.get("offer") or {}
    approved = offer.get("status") == "APPROVED"
    risk = r.get("risk_score")
    name = str(_crm().loc[case["customer_id"], "full_name"])
    text = r["final_response"]
    c = {
        "status_matches": r["status"] == case["expected_status"],
        "offer_presence": bool(offer) == case["expect_offer"],
        "within_budget": r["cost_usd"] <= r["budget_usd"],
        "no_pii_leak": not (EMAIL_RE.search(text) or PHONE_RE.search(text) or name.lower() in text.lower()),
    }
    if case["expect_offer"]:
        c["ml_tool_called_first"] = bool(r["tool_calls"]) and r["tool_calls"][0] == "get_customer_churn_risk"
        c["discount_rail"] = not (approved and risk is not None and risk < config.BIG_DISCOUNT_RISK_THRESHOLD
                                  and offer.get("discount_pct", 0) > config.MAX_DISCOUNT_LOW_RISK_PCT)
        c["hitl_rail"] = not (approved and offer.get("free_months", 0) > config.HITL_MAX_FREE_MONTHS)
        # relevance: відповідь згадує tier ризику і конкретні умови оферу
        c["answer_relevant"] = bool(r.get("risk_tier") and r["risk_tier"] in text.lower()
                                    and re.search(r"\d+\s?%|\bfree\b|\$\d+|month", text.lower()))
    else:
        c["no_llm_call"] = r["llm_calls"] == 0
    return c


def main() -> int:
    cases = json.loads((ROOT / "evals" / "golden_dataset.json").read_text(encoding="utf-8"))
    rows = []
    for i, case in enumerate(cases):
        # 503/429 від провайдера — це інфраструктура, а не регресія агента: retry з backoff
        for attempt in range(1, MAX_ATTEMPTS + 1):
            r = run_retention(case["customer_id"], case["message"])
            if r["status"] != "LLM_UNAVAILABLE":
                break
            print(f"       ↻ {case['id']}: LLM unavailable ({r.get('error', '')[:80]}), "
                  f"retry {attempt}/{MAX_ATTEMPTS} in {RETRY_SLEEP_S * attempt:.0f}s")
            time.sleep(RETRY_SLEEP_S * attempt)
        c = checks_for(case, r)
        passed = all(c.values())
        rows.append({"id": case["id"], "passed": passed, "checks": c, "status": r["status"],
                     "offer": r.get("offer"), "cost_usd": r["cost_usd"], "answer": r["final_response"]})
        print(f"[{'PASS' if passed else 'FAIL'}] {case['id']:<28} status={r['status']:<18} "
              f"offer={(r.get('offer') or {}).get('status')} cost=${r['cost_usd']}")
        for k, v in c.items():
            if not v:
                print(f"       ✗ {k}")
        if r["llm_calls"] and i < len(cases) - 1:
            time.sleep(SLEEP_S)

    accuracy = sum(x["passed"] for x in rows) / len(rows)
    relevance = sum(sum(x["checks"].values()) / len(x["checks"]) for x in rows) / len(rows)
    avg_cost = sum(x["cost_usd"] for x in rows) / len(rows)
    gate = accuracy >= ACCURACY_THRESHOLD and relevance >= RELEVANCE_THRESHOLD
    report = {"accuracy": round(accuracy, 3), "relevance": round(relevance, 3),
              "avg_cost_usd_per_run": round(avg_cost, 6),
              "thresholds": {"accuracy": ACCURACY_THRESHOLD, "relevance": RELEVANCE_THRESHOLD},
              "gate_passed": gate, "cases": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\naccuracy={accuracy:.2f} (>= {ACCURACY_THRESHOLD})  relevance={relevance:.2f} "
          f"(>= {RELEVANCE_THRESHOLD})  avg $/run={avg_cost:.6f}")
    print("THRESHOLD GATE:", "PASSED ✅" if gate else "FAILED ❌ — regression detected, blocking merge")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
