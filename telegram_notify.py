"""
Telegram notification module for the crypto pipeline.
Reads credentials from .env and sends daily signal summaries.
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BINANCE_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠  Telegram not configured")
        return
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }, timeout=10)
    if resp.status_code != 200:
        print(f"  ⚠  Telegram error: {resp.text}")


def notify_signals(signals: list):
    if not signals:
        return

    lines = ["<b>📊 Daily Crypto Signals</b>"]
    lines.append(f"<i>{signals[0].get('date', '')}</i>")
    lines.append("")

    for s in signals:
        coin      = s.get("coin", "")
        signal    = s.get("signal", "")
        prob_up   = s.get("prob_up", 0)
        kelly     = s.get("kelly_pct", 0)
        mtf       = s.get("mtf_score", 0)
        conf      = s.get("confidence", "")
        price     = s.get("price", 0)

        if "BUY" in signal:
            icon = "🟢"
        elif "OUT" in signal:
            icon = "🔴"
        else:
            icon = "⚪"

        lines.append(f"{icon} <b>{coin}</b> — {signal}")
        lines.append(f"   Price: <b>${price:,.2f}</b>")
        lines.append(f"   Confidence: {prob_up:.1f}% UP  [{conf}]")
        lines.append(f"   MTF: {mtf}/3 timeframes bullish")
        if kelly > 0:
            lines.append(f"   Position: <b>{kelly:.1f}% of capital</b>")
        lines.append("")

    # Market context
    fg      = signals[0].get("fear_greed", "")
    fg_lbl  = signals[0].get("fg_label", "")
    funding = signals[0].get("funding", "")
    btc_dom = signals[0].get("btc_dom", "")

    if fg:
        lines.append(f"😨 Fear &amp; Greed: <b>{fg}</b> ({fg_lbl})")
    if funding:
        lines.append(f"💰 Funding Rate: {float(funding):.4f}%")
    if btc_dom:
        lines.append(f"₿ BTC Dominance: {float(btc_dom):.1f}%")

    lines.append("")
    lines.append("<i>⚠ Model signals only — not financial advice.</i>")

    send_message("\n".join(lines))
    print("  ✓  Telegram notification sent")
