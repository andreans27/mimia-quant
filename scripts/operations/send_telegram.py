#!/usr/bin/env python3
"""Send hourly report to Telegram."""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv('/root/projects/mimia-quant/.env')
token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

msg = (
    "Mimia Hourly Report - 2026-04-25 15:05 UTC\n"
    "\n"
    "RETRAINED (2 symbols):\n"
    "  FETUSDT - OK (deployed)\n"
    "  TIAUSDT - OK (deployed)\n"
    "\n"
    "LIVE STATUS: Balance $4361.97\n"
    "Open: UNIUSDT (long)\n"
    "\n"
    "KELLY SIZING: All symbols < 30 trades - skipped\n"
    "\n"
    "RETRAIN STATUS:\n"
    "  APTUSDT   WR60% PF1.83 [14:18]\n"
    "  BTCUSDT   WR62% PF4.37 [15:00]\n"
    "  FETUSDT   WR83% PF12.05 [15:03]\n"
    "  TIAUSDT   WR81% PF11.59 [15:05]\n"
    "  UNIUSDT   WR55% PF1.64 [14:20]\n"
    "\n"
    "Cycle: 238s | No errors"
)

r = requests.post(
    f'https://api.telegram.org/bot{token}/sendMessage',
    json={'chat_id': chat_id, 'text': msg},
    timeout=15
)
print(f"Status: {r.status_code}")
print(f"OK: {r.json().get('ok')}")
sys.exit(0 if r.json().get('ok') else 1)
