#!/usr/bin/env python3
"""
Database initialization script for Mimia Quant Trading System.
Creates all tables and optionally seeds sample data.

Usage:
    python scripts/init_db.py [--drop] [--seed]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.database import (
    Database,
    MarketBars,
    Trades,
    EquityCurve,
    StrategyPerformance,
    ParametersLog,
    FundingRates,
    OrderLog,
)


def main():
    parser = argparse.ArgumentParser(description="Initialize Mimia Quant database")
    parser.add_argument("--drop", action="store_true", help="Drop all tables first")
    parser.add_argument("--seed", action="store_true", help="Seed with sample data")
    parser.add_argument("--path", type=str, default="data/mimia_quant.db", help="DB path")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    db_path = project_root / args.path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MIMIA QUANT - DATABASE INITIALIZATION")
    print(f"{'='*60}")
    print(f"Database: {db_path}")

    db = Database(db_path=str(db_path))

    if args.drop:
        print("\nDropping existing tables...")
        db.drop_all()
        print("Tables dropped.")

    print("\nCreating tables...")
    db.create_all()

    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"Tables created: {', '.join(tables)}")

    if args.seed:
        print("\nSeeding sample data...")
        seed_database(db)

    print(f"\n{'='*60}")
    print("INITIALIZATION COMPLETE")
    print(f"{'='*60}\n")


def seed_database(db: Database):
    """Seed with minimal sample data for testing."""
    import random
    from datetime import datetime, timedelta

    with db.get_session() as session:
        # Sample market bars
        base_time = datetime.utcnow() - timedelta(hours=100)
        base_price = 50000.0
        for i in range(100):
            t = base_time + timedelta(hours=i)
            close = base_price + random.uniform(-500, 500)
            bar = MarketBars(
                symbol="BTCUSDT",
                timeframe="1h",
                timestamp=t,
                open=base_price,
                high=max(base_price, close) + random.uniform(0, 100),
                low=min(base_price, close) - random.uniform(0, 100),
                close=close,
                volume=random.uniform(100, 1000),
            )
            session.add(bar)
            base_price = close

        # Sample trade
        trade = Trades(
            trade_id="sample_001",
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            quote_quantity=5000.0,
            commission=2.0,
            executed_at=datetime.utcnow() - timedelta(days=1),
        )
        session.add(trade)

        # Sample equity
        equity = EquityCurve(
            timestamp=datetime.utcnow() - timedelta(days=1),
            equity=5000.0,
            cash=5000.0,
            total_value=5000.0,
            daily_pnl=50.0,
            daily_return=0.01,
            cumulative_return=0.0,
            drawdown=0.0,
            strategy_name="momentum",
            session_id="main",
        )
        session.add(equity)

        session.commit()
        print("Sample data seeded.")


if __name__ == "__main__":
    main()
