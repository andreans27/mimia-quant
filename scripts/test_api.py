#!/usr/bin/env python3
"""Test Binance API connection with testnet credentials."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from utils.binance_client import BinanceRESTClient


def main():
    print("Testing Binance Testnet API connection...")
    print(f"API Key: {os.environ['BINANCE_TESTNET_API_KEY'][:10]}...")

    client = BinanceRESTClient(
        testnet=True,
        testnet_api_key=os.getenv("BINANCE_TESTNET_API_KEY", ""),
        testnet_api_secret=os.getenv("BINANCE_TESTNET_API_SECRET", ""),
    )

    # Test account info
    print("\n1. Account Info:")
    try:
        account = client.get_account_info()
        print(f"   Status: OK")
        print(f"   Account Type: {account.get('accountType', 'N/A')}")
        if 'assets' in account:
            for b in account['assets'][:3]:
                if float(b.get('availableBalance', 0)) > 0:
                    print(f"   {b['asset']}: available={b['availableBalance']}, marginBalance={b.get('marginBalance', 'N/A')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test balance
    print("\n2. Account Balance:")
    try:
        balances = client.get_balance()
        print(f"   Status: OK")
        for b in balances[:5]:
            if float(b.get('balance', 0)) > 0:
                print(f"   {b['asset']}: {b['balance']}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test orderbook
    print("\n3. BTCUSDT Orderbook (5 levels):")
    try:
        ob = client.get_orderbook_depth("BTCUSDT", limit=5)
        print(f"   Status: OK")
        print(f"   Bids: {len(ob.get('bids', []))} levels")
        print(f"   Asks: {len(ob.get('asks', []))} levels")
        if ob.get('bids'):
            print(f"   Top bid: {ob['bids'][0]}")
        if ob.get('asks'):
            print(f"   Top ask: {ob['asks'][0]}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test funding rate
    print("\n4. BTCUSDT Funding Rate:")
    try:
        fr = client.get_funding_rate("BTCUSDT")
        print(f"   Status: OK")
        print(f"   Funding Rate: {fr}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test recent trades
    print("\n5. BTCUSDT Recent Trades:")
    try:
        trades = client.get_recent_trades("BTCUSDT", limit=5)
        print(f"   Status: OK")
        print(f"   Count: {len(trades)}")
        if trades:
            t = trades[0]
            print(f"   Latest: price={t.get('price')}, qty={t.get('qty')}, time={t.get('time')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test klines
    print("\n6. BTCUSDT Kline (5 x 1h):")
    try:
        klines = client.get_klines("BTCUSDT", interval="1h", limit=5)
        print(f"   Status: OK")
        print(f"   Count: {len(klines)}")
        if klines:
            k = klines[-1]
            print(f"   Latest: open={k[1]}, high={k[2]}, low={k[3]}, close={k[4]}, vol={k[5]}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test mark price
    print("\n7. BTCUSDT Mark Price:")
    try:
        mp = client.get_mark_price("BTCUSDT")
        print(f"   Status: OK")
        print(f"   Mark Price: {mp}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "="*50)
    print("API CONNECTION TEST COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
