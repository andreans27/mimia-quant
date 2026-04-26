#!/usr/bin/env python3
"""Send hourly report to Telegram."""
from dotenv import load_dotenv
import os
import sys
import requests

load_dotenv('.env')
token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '766684679')

if not token:
    print("ERROR: TELEGRAM_BOT_TOKEN not set")
    sys.exit(1)

report_path = sys.argv[1] if len(sys.argv) > 1 else 'data/cron_output/report_20260426_044623.txt'

with open(report_path, 'r') as f:
    report = f.read()

if len(report) > 3500:
    report = report[:3497] + '...'

url = f'https://api.telegram.org/bot{token}/sendMessage'
resp = requests.post(url, json={
    'chat_id': chat_id,
    'text': report,
    'parse_mode': 'Markdown'
}, timeout=15)

print(f'Status: {resp.status_code}')
print(f'Response: {resp.text[:300]}')

if resp.status_code != 200:
    sys.exit(1)
