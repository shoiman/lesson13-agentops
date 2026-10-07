"""Бізнес-правила (rails) для retention-оферу. Чистий детермінований код — тестується без LLM."""
from dataclasses import dataclass, asdict

from agent import config


@dataclass
class OfferDecision:
    status: str            # APPROVED | REJECTED | PENDING_HUMAN_REVIEW
    reasons: list
    discount_pct: float
    free_months: int
    bonus_value_usd: float

    def to_dict(self):
        return asdict(self)


def evaluate_offer(risk_score: float, monthly_charges: float, discount_pct: float,
                   free_months: int, bonus_value_usd: float) -> OfferDecision:
    reasons = []
    # Rail 1: знижка > 20% тільки для high-risk клієнтів
    if discount_pct > config.MAX_DISCOUNT_LOW_RISK_PCT and risk_score < config.BIG_DISCOUNT_RISK_THRESHOLD:
        reasons.append(f"discount {discount_pct}% > {config.MAX_DISCOUNT_LOW_RISK_PCT}% "
                       f"is forbidden for risk_score {risk_score:.2f} < {config.BIG_DISCOUNT_RISK_THRESHOLD}")
    # Rail 2: абсолютна стеля
    if discount_pct > config.MAX_DISCOUNT_ABSOLUTE_PCT:
        reasons.append(f"discount {discount_pct}% exceeds absolute cap {config.MAX_DISCOUNT_ABSOLUTE_PCT}%")
    if discount_pct < 0 or free_months < 0 or bonus_value_usd < 0:
        reasons.append("negative offer values are invalid")
    if reasons:
        return OfferDecision("REJECTED", reasons, discount_pct, free_months, bonus_value_usd)

    # Human-in-the-loop: дорогі рішення → ручна перевірка
    total_bonus = bonus_value_usd + free_months * monthly_charges
    hitl = []
    if free_months > config.HITL_MAX_FREE_MONTHS:
        hitl.append(f"free period {free_months} months > {config.HITL_MAX_FREE_MONTHS}")
    if total_bonus > config.HITL_MAX_BONUS_USD:
        hitl.append(f"total bonus value ${total_bonus:.2f} > ${config.HITL_MAX_BONUS_USD}")
    if hitl:
        return OfferDecision("PENDING_HUMAN_REVIEW", hitl, discount_pct, free_months, bonus_value_usd)
    return OfferDecision("APPROVED", ["within policy"], discount_pct, free_months, bonus_value_usd)
