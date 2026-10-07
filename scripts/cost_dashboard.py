"""Cost dashboard (Lab 1): $/run і latency кожного span-у з Phoenix → results/cost_dashboard.png.

  python scripts/cost_dashboard.py
Дані беруться з Phoenix (http://localhost:6006) через phoenix.client; ціна — з .env (PRICE_*_PER_1M).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from phoenix.client import Client  # noqa: E402

from agent import config  # noqa: E402

client = Client(base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
df = client.spans.get_spans_dataframe(project_identifier=config.PHOENIX_PROJECT_NAME, limit=5000)
if df is None or df.empty:
    sys.exit("No spans found — run the agent first (python run_agent.py --demo)")

df["latency_ms"] = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds() * 1000
col = lambda c: df[c] if c in df.columns else pd.Series(0, index=df.index)  # noqa: E731
df["tok_in"] = col("attributes.llm.token_count.prompt").fillna(0)
df["tok_out"] = col("attributes.llm.token_count.completion").fillna(0)
df["cost"] = (df["tok_in"] * config.PRICE_INPUT_PER_1M + df["tok_out"] * config.PRICE_OUTPUT_PER_1M) / 1e6

roots = df[df["name"] == "retention_agent_run"].copy()
if roots.empty:
    sys.exit("No retention_agent_run traces yet")
cost_per_trace = df.groupby("context.trace_id")["cost"].sum()
roots["run_cost"] = roots["context.trace_id"].map(cost_per_trace).fillna(0)


def agent_attr(row, key):
    """Атрибут agent.<key>: Phoenix віддає його або плоскою колонкою, або вкладеним dict."""
    flat = row.get(f"attributes.agent.{key}")
    if flat is not None and flat == flat:
        return flat
    nested = row.get("attributes.agent")
    return nested.get(key, "?") if isinstance(nested, dict) else "?"


roots["label"] = [f"{agent_attr(r, 'customer_id')}\n{agent_attr(r, 'status')}" for _, r in roots.iterrows()]
roots = roots.sort_values("start_time")

kinds = df["span_kind"].astype(str).str.upper()
span_lat = (df[kinds.isin(["LLM", "TOOL", "GUARDRAIL", "AGENT"])]
            .groupby("name")["latency_ms"].agg(["mean", "max", "count"]).sort_values("mean"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
ax.bar(range(len(roots)), roots["run_cost"], color="#14b8a6")
ax.axhline(roots["run_cost"].mean(), color="#ef4444", ls="--", label=f"avg \\$/run = \\${roots['run_cost'].mean():.5f}")
ax.set_xticks(range(len(roots)), roots["label"], fontsize=8, rotation=30, ha="right")
ax.set_ylabel("USD per run")
ax.set_title(f"Cost per agent run ({len(roots)} runs, {config.GEMINI_MODEL})")
ax.legend()
ax = axes[1]
ax.barh(span_lat.index, span_lat["mean"], color="#1e3a5f", label="mean")
ax.scatter(span_lat["max"], span_lat.index, color="#f59e0b", zorder=3, label="max")
ax.set_xlabel("latency, ms")
ax.set_title("Latency per span (LLM / tools / guardrails)")
ax.legend()
fig.suptitle(f"Telco Retention Agent — AgentOps cost dashboard | total \\${roots['run_cost'].sum():.5f}, "
             f"p50 run latency {roots['latency_ms'].median():.0f} ms", fontsize=12)
fig.tight_layout()
(ROOT / "results").mkdir(exist_ok=True)
out = ROOT / "results" / "cost_dashboard.png"
fig.savefig(out, dpi=130)
print(roots[["label", "run_cost", "latency_ms"]].to_string(index=False))
print(f"\navg $/run = {roots['run_cost'].mean():.6f}\nsaved {out}")
