"""Slack-алерти (Incoming Webhook). Без SLACK_WEBHOOK_URL — dry-run у консоль + alerts.log."""
import datetime as dt
import json
import requests

from agent import config


def send_alert(title: str, details: dict) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    text = f":rotating_light: *{title}*\n```{json.dumps(details, ensure_ascii=False, indent=2)}```"
    with open("alerts.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "title": title, **details}, ensure_ascii=False) + "\n")
    if not config.SLACK_WEBHOOK_URL:
        print(f"[SLACK-DRY-RUN] {title}: {details}")
        return
    try:
        r = requests.post(config.SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        print(f"[SLACK] {title} -> HTTP {r.status_code}")
    except requests.RequestException as e:  # алерт не має ламати агента
        print(f"[SLACK-ERROR] {e}")
