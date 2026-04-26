#!/usr/bin/env python3
"""Send the latest cron report to Telegram."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv('.env')

import os
import requests

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '766684679')

if not token:
    print("ERROR: TELEGRAM_BOT_TOKEN not found")
    sys.exit(1)

# Read the latest report
report_path = Path('data/cron_output/report_20260426_105139.txt')
if not report_path.exists():
    print(f"ERROR: Report not found at {report_path}")
    sys.exit(1)

report = report_path.read_text()

# Truncate if > 3500 chars
if len(report) > 3500:
    report = report[:3497] + '...'

url = f'https://api.telegram.org/bot{token}/sendMessage'
response = requests.post(url, json={
    'chat_id': chat_id,
    'text': report,
    'parse_mode': 'Markdown'
}, timeout=15)

print(f'Status: {response.status_code}')
resp_data = response.json()
if resp_data.get('ok'):
    print('Telegram message sent successfully!')
else:
    print(f'Telegram error: {resp_data.get("description", "unknown")}')
