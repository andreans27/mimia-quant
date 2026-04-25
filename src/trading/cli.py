#!/usr/bin/env python3
"""
Mimia Quant - Live Trading CLI
===============================
Command-line interface for the live trading system.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import traceback

from dotenv import load_dotenv
load_dotenv()  # Load .env for Binance API keys

from src.trading.state import init_db, reset_state, DB_PATH
from src.trading.engine import LiveTrader
from src.trading.reporter import show_status, show_report


def main():
    """Main CLI entry point for live trading."""
    parser = argparse.ArgumentParser(description='Mimia Live Trading Engine')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--report', action='store_true', help='Send daily report')
    parser.add_argument('--reset', action='store_true', help='Reset trading state')
    # Network mode: testnet (default) or mainnet
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--testnet', action='store_true', default=False,
                           help='Use Binance testnet (default)')
    mode_group.add_argument('--mainnet', action='store_true', default=False,
                           help='⚠️ Use Binance MAINNET — real funds at risk!')
    args = parser.parse_args()

    if args.init:
        init_db()
        print(f"✅ Database initialized at {DB_PATH}")
    elif args.status:
        show_status()
    elif args.report:
        show_report(testnet=not args.mainnet)
    elif args.reset:
        reset_state()
    else:
        # Determine testnet mode: --mainnet overrides default
        use_testnet = not args.mainnet
        network_label = "TESTNET" if use_testnet else "⚠️ MAINNET (REAL FUNDS) ⚠️"
        print(f"\n{'='*60}")
        print(f"  Mode: {network_label}")
        print(f"{'='*60}")
        # Run one live trading cycle
        pt = LiveTrader(testnet=use_testnet)
        try:
            pt.run()
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        finally:
            pt.close()


if __name__ == '__main__':
    main()
