#!/usr/bin/env python3
"""Send the hourly report to Telegram."""
from dotenv import load_dotenv
import os, requests

load_dotenv('/root/projects/mimia-quant/.env')
token = os.environ['TELEGRAM_BOT_TOKEN']
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '766684679')

with open('/root/projects/mimia-quant/data/cron_output/report_20260426_064801.txt', 'r') as f:
    report = f.read()

if len(report) > 3500:
    report = report[:3497] + '...'

url = f'https://api.telegram.org/bot{token}/sendMessage'
payload = {
    'chat_id': chat_id,
    'text': report,
    'parse_mode': 'Markdown'
}
resp = requests.post(url, json=payload, timeout=15)
print(f'Status: {resp.status_code}')
print(f'Response: {resp.text[:300]}')
