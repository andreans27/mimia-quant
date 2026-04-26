#!/usr/bin/env python3
"""Cron runner for mimia-quant hourly pipeline with Telegram delivery."""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
BASE_DIR = Path('/root/projects/mimia-quant')
os.chdir(str(BASE_DIR))
load_dotenv(BASE_DIR / '.env')

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '766684679')

def send_telegram(text, parse_mode='Markdown'):
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not found in .env", file=sys.stderr)
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if not resp.ok:
            print(f"Telegram API error: {resp.status_code} {resp.text}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"Telegram send failed: {e}", file=sys.stderr)
        return False

def run_pipeline():
    """Run the hourly cron pipeline."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting hourly pipeline...")
    
    result = subprocess.run(
        ['python', 'scripts/operations/cron_hourly.py'],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(BASE_DIR)
    )
    
    stdout = result.stdout or ''
    stderr = result.stderr or ''
    returncode = result.returncode
    
    print(f"Return code: {returncode}")
    if stdout:
        print(f"STDOUT ({len(stdout)} chars):")
        print(stdout[-3000:] if len(stdout) > 3000 else stdout)
    if stderr:
        print(f"STDERR ({len(stderr)} chars):")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
    
    return stdout, stderr, returncode

def get_latest_cron_output():
    """Get report from latest file in data/cron_output/."""
    cron_dir = BASE_DIR / 'data' / 'cron_output'
    if not cron_dir.exists():
        return None
    files = sorted(cron_dir.glob('*.txt'), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    latest = files[0]
    content = latest.read_text(encoding='utf-8', errors='replace')
    print(f"Found cron output file: {latest.name} ({len(content)} chars)")
    return content

def format_report(stdout, stderr, returncode):
    """Format the report from pipeline output."""
    # Try to get report from cron_output directory first
    cron_report = get_latest_cron_output()
    if cron_report and cron_report.strip():
        report = cron_report.strip()
    elif stdout.strip():
        report = stdout.strip()
    else:
        report = stderr.strip() if stderr.strip() else "Pipeline selesai tanpa output."
    
    # Truncate if too long for Telegram (3500 chars)
    if len(report) > 3500:
        report = report[:3497] + "..."
    
    return report

def main():
    # Need to import requests here after load_dotenv might modify path
    global requests
    import requests
    
    try:
        stdout, stderr, returncode = run_pipeline()
    except subprocess.TimeoutExpired:
        error_msg = "⚠️ *Mimia Cron Hourly — TIMEOUT*\n\nPipeline melebihi batas 900 detik. Proses dihentikan."
        print(error_msg)
        send_telegram(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"⚠️ *Mimia Cron Hourly — ERROR*\n\nPipeline gagal: {e}"
        print(error_msg)
        send_telegram(error_msg)
        sys.exit(1)
    
    # Check for errors
    if returncode != 0:
        error_detail = stderr.strip()[-500:] if stderr.strip() else "No stderr"
        error_msg = (
            f"⚠️ *Mimia Cron Hourly — ERROR (exit {returncode})*\n\n"
            f"```\n{error_detail}\n```"
        )
        if len(error_msg) > 3500:
            error_msg = error_msg[:3497] + "..."
        print(error_msg)
        send_telegram(error_msg)
        sys.exit(1)
    
    # Format and send report
    report = format_report(stdout, stderr, returncode)
    
    if not report.strip():
        report = "[SILENT]"
        print("No output from pipeline — sending [SILENT]")
        # Don't send anything for silent
        return
    
    if report == "[SILENT]":
        print("Report is [SILENT] — suppressing delivery.")
        return
    
    # Add header
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S WIB')
    full_report = f"🤖 *Mimia Cron Hourly*\n{timestamp}\n\n{report}"
    
    if len(full_report) > 3500:
        full_report = full_report[:3497] + "..."
    
    success = send_telegram(full_report)
    if success:
        print("Report sent to Telegram successfully.")
    else:
        print("Failed to send report to Telegram.")

if __name__ == '__main__':
    main()
