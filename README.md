# Lesson 13 — AgentOps & Autonomous Systems Reliability

**Telco-Churn Retention Agent**: LangGraph-агент (Gemini), який за ML-прогнозом churn-ризику
пропонує клієнту retention-офер (знижка / бонус) — з повною трасуванням в **Arize Phoenix**,
guardrails, SLA-бюджетами, circuit breaker, human-in-the-loop та eval-gate у GitHub Actions.

> Post-mortem (Part 3) у цій роботі свідомо не виконувався.

## Архітектура

```
customer message ─► guardrail.input (Guardrails AI: prompt injection) ──blocked──► відмова, 0 LLM-викликів
                          │ pass
                          ▼
              LangGraph ReAct agent (Gemini)  ◄── BudgetMiddleware (hard-stop $, soft warn 80%, max 6 steps)
               ├─ get_customer_churn_risk     ML-модель → score, tier, бюджет на reasoning   ┐
               ├─ get_customer_profile        CRM (містить PII!)                             │ circuit breaker
               ├─ search_retention_playbook   allowlist доменів (Guardrails AI)              │ на кожен tool
               └─ propose_retention_offer     policy rails + HITL-ескалація                  ┘
                          ▼
              guardrail.output (Guardrails AI: PII redaction) ─► відповідь
Все це — один trace у Phoenix (OpenTelemetry / OpenInference).
```

## Що де реалізовано

| Вимога | Реалізація |
|---|---|
| **Lab 1** Tracing + cost dashboard | `agent/tracing.py` (Phoenix + `LangChainInstrumentor`), root span `retention_agent_run` з атрибутами `agent.cost_usd/budget_usd/risk_tier/status`; `scripts/cost_dashboard.py` → `results/cost_dashboard.png` (avg $/run + latency кожного span) |
| **Lab 2** Guardrails | `agent/guards.py`: власні Guardrails AI валідатори `RedactPII` (on_fail=FIX), `DetectPromptInjection` (EXCEPTION), `AllowedDomain` (EXCEPTION) для search tool |
| **Lab 3** SLA | `agent/sla.py`: `BudgetMiddleware` (LangChain callback, `raise_error=True` → hard-stop, дефолт $0.15/run), step limit 6, `CircuitBreaker` CLOSED→OPEN→HALF_OPEN; `agent/alerts.py` — Slack Incoming Webhook (без `SLACK_WEBHOOK_URL` → dry-run у консоль + `alerts.log`) |
| **Lab 4** Regression | `evals/golden_dataset.json` (7 кейсів), `evals/run_evals.py` — rule-based judge + Threshold Gate (accuracy ≥ 0.85, relevance ≥ 0.90); `.github/workflows/agent-evals.yml` |
| **Part 2** ML tool | `scripts/train_model.py` — HistGradientBoosting на `telco_customers.csv` (25k рядків, ROC-AUC 0.73), `get_customer_churn_risk` у `agent/tools.py` |
| Бюджет за ризиком | `agent/config.py`: high (≥0.70) **$0.50**, medium (≥0.40) **$0.15**, low **$0.05** — встановлюється після ML-прогнозу |
| Rail на знижку | `agent/policy.py`: знижка > 20% заборонена, якщо risk < 0.70; абсолютна стеля 40% |
| Human-in-the-loop | > 3 безкоштовних місяців **або** сумарний бонус > $500 → `PENDING_HUMAN_REVIEW`, запис у `escalations.jsonl` + Slack-алерт |
| Red teaming | G6 — пряма prompt injection (блокується до LLM), G7 — соціальна інженерія «вже погодили 50%» (проходить фільтр, але policy rail не дає > 20%) |

Чому власна компактна модель, а не `churn_model.pkl` з telco-churn-mlops-synthetic-08: той файл ~190 МБ
(ліміт GitHub — 100 МБ), а CI має запускати агента разом з моделлю. Датасет і фічі ті самі; модель ~150 КБ.
`data/customers_crm.csv` — 300 тестових клієнтів з **синтетичними** ПІБ/email/телефоном для перевірки PII-guardrail.

## Запуск

```bash
python3 -m venv ~/venvs/lesson13 && source ~/venvs/lesson13/bin/activate
pip install -r requirements.txt
cp .env.example .env            # GOOGLE_API_KEY, GEMINI_MODEL
python scripts/train_model.py "../telco-churn-mlops-synthetic-08/data/telco_customers.csv"

phoenix serve                   # окремий термінал → http://localhost:6006
python run_agent.py --demo      # 5 сценаріїв: high / medium / low / HITL / injection
python scripts/cost_dashboard.py
python scripts/demo_circuit_breaker.py
python run_agent.py --customer 4097-BGZQO --budget 0.0001   # hard-stop
python -m pytest -q             # 18 тестів без LLM
python evals/run_evals.py       # golden dataset + threshold gate
```

## Результати

| Сценарій | Ризик | Бюджет | Результат |
|---|---|---|---|
| 4097-BGZQO «too expensive» | 0.961 high | $0.50 | 30% → APPROVED |
| 4429-WVSXA «slow internet» | 0.497 medium | $0.15 | 15% → APPROVED |
| 9380-MEDIR «loyalty offers» | 0.309 low | $0.05 | 10% + 1 міс → APPROVED |
| 4097-BGZQO «6 free months» | 0.961 high | $0.50 | 35% + 6 міс ($981) → **PENDING_HUMAN_REVIEW** |
| 9380-MEDIR prompt injection | — | — | **BLOCKED_INJECTION**, 0 LLM-викликів |
| `--budget 0.0001` | — | $0.0001 | **BUDGET_EXCEEDED** після 1-го LLM-виклику |

- Середня вартість: **≈ $0.0012 / run** (gemini-3.1-flash-lite), p50 latency ≈ 7 с, LLM — найдовші span-и (~1.6 с).
- Circuit breaker: 3 помилки поспіль → OPEN + алерт → виклики блокуються → через timeout HALF_OPEN → CLOSED.
- Eval gate: **7/7 PASS, accuracy 1.00, relevance 1.00 → PASSED**.

Скріншоти (Phoenix trace tree, Metrics, cost dashboard, alerts, CI) — у звіті до домашнього завдання.
