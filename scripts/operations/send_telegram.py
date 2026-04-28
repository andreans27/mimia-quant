#!/usr/bin/env python3
"""Send cron report to Telegram."""
import requests, os, sys
from dotenv import load_dotenv

load_dotenv('.env')

token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '766684679')

if not token:
    print("ERROR: TELEGRAM_BOT_TOKEN not found", file=sys.stderr)
    sys.exit(1)

report_path = 'data/cron_output/report_20260427_072707.txt'
try:
    with open(report_path) as f:
        text = f.read()
except FileNotFoundError:
    text = "⚠️ Report file not found."
    sys.exit(1)

# Truncate if > 3500 chars
if len(text) > 3500:
    text = text[:3500] + '...\n_(truncated)_'

url = f'https://api.telegram.org/bot{token}/sendMessage'
payload = {
    'chat_id': chat_id,
    'text': text,
    'parse_mode': 'Markdown'
}
resp = requests.post(url, json=payload, timeout=15)
print(f'Status: {resp.status_code}')
print(f'Response: {resp.text[:300]}')
if resp.status_code != 200:
    sys.exit(1)
