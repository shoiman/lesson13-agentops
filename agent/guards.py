"""Guardrails AI: власні валідатори (без Guardrails Hub — не потрібен токен hub-у).

- RedactPII            — output guard: прибирає email / телефон / ПІБ клієнта з фінальної відповіді (on_fail=FIX)
- DetectPromptInjection — input guard: блокує спроби перехопити інструкції агента (on_fail=EXCEPTION)
- AllowedDomain        — tool guard: search tool може ходити лише на allowlist доменів (on_fail=EXCEPTION)
"""
import re
import warnings
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from guardrails import Guard, OnFailAction
from guardrails.validators import (FailResult, PassResult, ValidationResult, Validator,
                                   register_validator)

from agent import config

# Guardrails у синхронному коді попереджає про відсутній event loop — це очікувано
warnings.filterwarnings("ignore", message="Could not obtain an event loop")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+\d{1,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (all |the )?(previous|prior|above|your) (instructions|rules|policy)",
    r"forget (all |your )?(previous )?(instructions|rules|policy)",
    r"you are now", r"new (system )?instructions?:", r"system prompt",
    r"developer mode", r"act as (the |a )?(admin|manager|supervisor)",
    r"(override|bypass) (the )?(policy|rules|limits|guardrails?)",
    r"(maximum|max|100\s?%) discount", r"\b(9\d|100)\s?% (off|discount)",
    r"ігноруй", r"игнорируй",
]


@register_validator(name="telco/redact_pii", data_type="string")
class RedactPII(Validator):
    def __init__(self, known_names: Optional[list] = None, on_fail: Optional[Callable] = None, **kw):
        super().__init__(on_fail=on_fail, known_names=known_names, **kw)
        self.known_names = [n for n in (known_names or []) if n]

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        fixed, found = str(value), []
        for rx, tag in ((EMAIL_RE, "[EMAIL_REDACTED]"), (PHONE_RE, "[PHONE_REDACTED]")):
            if rx.search(fixed):
                found.append(tag)
                fixed = rx.sub(tag, fixed)
        for name in self.known_names:
            if name.lower() in fixed.lower():
                found.append("[NAME_REDACTED]")
                fixed = re.sub(re.escape(name), "[NAME_REDACTED]", fixed, flags=re.I)
        if found:
            return FailResult(error_message=f"PII detected: {sorted(set(found))}", fix_value=fixed)
        return PassResult()


@register_validator(name="telco/detect_prompt_injection", data_type="string")
class DetectPromptInjection(Validator):
    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        text = str(value).lower()
        hits = [p for p in INJECTION_PATTERNS if re.search(p, text)]
        if hits:
            return FailResult(error_message=f"Prompt injection detected (patterns: {hits[:3]})")
        return PassResult()


@register_validator(name="telco/allowed_domain", data_type="string")
class AllowedDomain(Validator):
    def __init__(self, allowed: Optional[list] = None, on_fail: Optional[Callable] = None, **kw):
        super().__init__(on_fail=on_fail, allowed=allowed, **kw)
        self.allowed = set(allowed or [])

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        host = (urlparse(str(value)).hostname or "").lower()
        if urlparse(str(value)).scheme != "https" or host not in self.allowed:
            return FailResult(error_message=f"Domain '{host or value}' is not in allowlist {sorted(self.allowed)}")
        return PassResult()


class GuardrailViolation(Exception):
    pass


def _guard(name: str, validator: Validator) -> Guard:
    g = Guard(name=name).use(validator)
    g.configure(allow_metrics_collection=False)   # не відправляти телеметрію Guardrails назовні
    return g


input_guard = _guard("input_guard", DetectPromptInjection(on_fail=OnFailAction.EXCEPTION))
domain_guard = _guard("domain_guard", AllowedDomain(allowed=sorted(config.ALLOWED_SEARCH_DOMAINS),
                                                    on_fail=OnFailAction.EXCEPTION))


def check_input(text: str) -> None:
    try:
        input_guard.validate(text)
    except Exception as e:
        raise GuardrailViolation(str(e)) from e


def check_domain(url: str) -> None:
    try:
        domain_guard.validate(url)
    except Exception as e:
        raise GuardrailViolation(str(e)) from e


def redact_output(text: str, known_names: list) -> tuple[str, bool]:
    """Повертає (безпечний текст, чи було знайдено PII)."""
    guard = _guard("output_pii_guard", RedactPII(known_names=known_names, on_fail=OnFailAction.FIX))
    outcome = guard.validate(text)
    safe = outcome.validated_output if outcome.validated_output is not None else text
    return safe, safe != text
