#!/usr/bin/env python3
"""Send hourly cron report to Telegram."""
from dotenv import load_dotenv; load_dotenv('.env')
import os, requests

report = (
    "📊 **Mimia Hourly Report**\n\n"
    "💰 **$4916.41** | 📊 🟢 ENAUSDT • 🟢 OPUSDT • 🟢 WIFUSDT • 🟢 DOGEUSDT • 🟢 SOLUSDT • 🟢 ARBUSDT • 🟢 LINKUSDT • 🟢 AAVEUSDT | 📈 Today: 97 trades | PnL $40.82\n\n"
    "**📐 Kelly Sizing**\n"
    "  `FETUSDT   ` 5%  WR37%  PF0.40  (30 tr)\n"
    "  `SEIUSDT   ` 5%  WR11%  PF2.31  (35 tr)\n\n"
    "**🔍 Signal Quality**\n```\n"
    "SYMBOL     LIVE-WR LIVE-PF  BT-WR ACTION   PRIO\n"
    "---------- ------- ------- ------ -------- ----\n"
    "1000PEPEUSDT     N/A     N/A    89% 🟢SKIP     9\n"
    "AAVEUSDT       21%    0.38    82% 🟢SKIP     5\n"
    "ADAUSDT        ~0%     N/A    87% 🟢SKIP     5\n"
    "APTUSDT        N/A     N/A    90% 🟢SKIP     9\n"
    "ARBUSDT        19%    0.81    87% 🟢SKIP     5\n"
    "AVAXUSDT       ~0%     N/A    83% 🟢SKIP     9\n"
    "BNBUSDT        ~0%     N/A    70% 🟢SKIP     5\n"
    "BTCUSDT        N/A     N/A    64% 🟢SKIP     5\n"
    "DOGEUSDT       ~0%     N/A    80% 🟢SKIP     5\n"
    "ENAUSDT        17%    0.17    87% 🟢SKIP     5\n"
    "ETHUSDT        50%    4.35    73% 🟢SKIP     6\n"
    "FETUSDT        ~0%     N/A    86% 🟢SKIP     9\n"
    "INJUSDT        ~0%     N/A    87% 🟢SKIP     5\n"
    "LINKUSDT       ~0%     N/A    81% 🟢SKIP     9\n"
    "NEARUSDT       ~0%     N/A    88% 🟢SKIP     5\n"
    "OPUSDT         ~0%     N/A    85% 🟢SKIP     5\n"
    "SEIUSDT        11%    2.31    82% 🟢SKIP     2\n"
    "SOLUSDT        29%    0.24    77% 🟢SKIP     5\n"
    "SUIUSDT        ~0%     N/A    87% 🟢SKIP     5\n"
    "TIAUSDT        29%    0.42    81% 🟢SKIP     5\n"
    "UNIUSDT        N/A     N/A    90% 🟢SKIP     9\n"
    "WIFUSDT        ~0%     N/A    81% 🟢SKIP     5\n"
    "WLDUSDT        ~0%     N/A    85% 🟢SKIP     5\n"
    "```\n\n"
    "**🔄 Retrain History**\n"
    "  `1000PEPEUSDT` WR89% PF22.05 [2026-04-26T06:03]\n"
    "  `AAVEUSDT  ` WR82% PF16.20 [2026-04-25T16:56]\n"
    "  `ADAUSDT   ` WR87% PF29.58 [2026-04-25T17:28]\n"
    "  `APTUSDT   ` WR90% PF31.62 [2026-04-26T14:58]\n"
    "  `ARBUSDT   ` WR87% PF29.06 [2026-04-25T17:03]\n"
    "  `AVAXUSDT  ` WR83% PF18.75 [2026-04-25T20:27]\n"
    "  `BNBUSDT   ` WR70% PF8.08 [2026-04-25T21:31]\n"
    "  `BTCUSDT   ` WR64% PF5.38 [2026-04-26T16:07]\n"
    "  `DOGEUSDT  ` WR80% PF13.87 [2026-04-25T16:36]\n"
    "  `ENAUSDT   ` WR87% PF27.32 [2026-04-25T16:37]\n"
    "  `ETHUSDT   ` WR73% PF11.20 [2026-04-25T22:36]\n"
    "  `FETUSDT   ` WR86% PF21.78 [2026-04-26T16:09]\n"
    "  `INJUSDT   ` WR87% PF27.04 [2026-04-25T18:04]\n"
    "  `LINKUSDT  ` WR81% PF13.09 [2026-04-25T23:42]\n"
    "  `NEARUSDT  ` WR88% PF29.14 [2026-04-25T17:29]\n"
    "  `OPUSDT    ` WR85% PF22.06 [2026-04-25T15:50]\n"
    "  `SEIUSDT   ` WR82% PF14.10 [2026-04-26T18:15]\n"
    "  `SOLUSDT   ` WR77% PF12.11 [2026-04-25T15:43]\n"
    "  `SUIUSDT   ` WR87% PF30.53 [2026-04-25T16:03]\n"
    "  `TIAUSDT   ` WR81% PF11.59 [2026-04-25T15:05]\n"
    "  `UNIUSDT   ` WR90% PF40.35 [2026-04-26T15:01]\n"
    "  `WIFUSDT   ` WR81% PF10.57 [2026-04-25T16:36]\n"
    "  `WLDUSDT   ` WR85% PF17.24 [2026-04-25T16:55]\n\n"
    "🤖 _Mimia — 2026-04-27 08:17 UTC_"
)

if len(report) > 3500:
    report = report[:3500] + "..."

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "766684679")
url = f"https://api.telegram.org/bot{token}/sendMessage"
resp = requests.post(url, json={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"}, timeout=15)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:200]}")
