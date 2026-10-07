"""Усі SLA / policy константи в одному місці."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.25"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "1.50"))

# ── SLA (Lab 3) ───────────────────────────────────────────────
DEFAULT_RUN_BUDGET_USD = 0.15     # hard-stop, поки ризик клієнта ще невідомий
SOFT_BUDGET_RATIO = 0.80          # soft warning на 80% бюджету
MAX_STEPS = 6                     # step limit (LLM-ітерації ReAct)
CB_FAILURE_THRESHOLD = 3          # circuit breaker: N поспіль помилок tool → OPEN
CB_RESET_TIMEOUT_S = 30           # OPEN → HALF_OPEN через N секунд

# ── Part 2: бюджет залежить від churn-ризику ──────────────────
RISK_TIERS = [                    # (нижня межа score, tier, бюджет $)
    (0.70, "high", 0.50),
    (0.40, "medium", 0.15),
    (0.00, "low", 0.05),
]

# ── Policy rails ─────────────────────────────────────────────
BIG_DISCOUNT_RISK_THRESHOLD = 0.70   # нижче цього score знижка > 20% заборонена
MAX_DISCOUNT_LOW_RISK_PCT = 20
MAX_DISCOUNT_ABSOLUTE_PCT = 40
HITL_MAX_FREE_MONTHS = 3             # > 3 безкоштовних місяців → ручна перевірка
HITL_MAX_BONUS_USD = 500             # загальна цінність бонусу > $500 → ручна перевірка

# ── Guardrails: дозволені домени для search tool ─────────────
ALLOWED_SEARCH_DOMAINS = {"kb.telco-retention.internal", "docs.telco-retention.internal"}

PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "telco-retention-agent")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def tier_for(score: float) -> tuple[str, float]:
    for lower, tier, budget in RISK_TIERS:
        if score >= lower:
            return tier, budget
    return "low", RISK_TIERS[-1][2]
