#!/usr/bin/env python3
"""
Database initialization script for Mimia Quant Trading System.

This script initializes the SQLite database and creates all required tables.
It can also seed the database with sample data for testing.

Usage:
    python scripts/init_db.py [--drop] [--seed] [--path <db_path>]

Options:
    --drop      Drop all existing tables before creating
    --seed      Seed the database with sample data
    --path      Path to database file (default: data/mimia_quant.db)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.database import (
    Database,
    EquityCurve,
    FundingRate,
    MarketBar,
    OrderLog,
    ParametersLog,
    StrategyPerformance,
    Trade,
    init_database,
)
from core.redis_client import RedisClient, get_redis_client


# =============================================================================
# SAMPLE DATA
# =============================================================================


def generate_sample_market_bars(db: Database, count: int = 100) -> None:
    """Generate sample market bar data."""
    print(f"Generating {count} sample market bars...")
    
    with db.get_session() as session:
        base_time = datetime.utcnow() - timedelta(hours=count)
        base_price = Decimal("50000.00")
        
        for i in range(count):
            open_time = base_time + timedelta(hours=i)
            close_time = open_time + timedelta(hours=1)
            
            # Random price movement
            change = Decimal(str(round((hash(str(i)) % 1000 - 500) / 100, 2)))
            open_price = base_price + change
            high_price = open_price + Decimal(str(round(abs(hash(str(i * 2)) % 500) / 100, 2)))
            low_price = open_price - Decimal(str(round(abs(hash(str(i * 3)) % 500) / 100, 2)))
            close_price = open_price + Decimal(str(round((hash(str(i * 4)) % 200 - 100) / 100, 2)))
            
            volume = Decimal(str(round(abs(hash(str(i * 5)) % 10000) / 100, 4)))
            quote_volume = volume * (open_price + close_price) / 2
            
            bar = MarketBar(
                symbol="BTCUSDT",
                exchange="binance",
                timeframe="1h",
                open_time=open_time,
                close_time=close_time,
                open_price=open_price,
                high_price=max(open_price, close_price, high_price),
                low_price=min(open_price, close_price, low_price),
                close_price=close_price,
                volume=volume,
                quote_volume=quote_volume,
                trades_count=int(abs(hash(str(i))) % 500) + 100,
                taker_buy_volume=volume * Decimal("0.45"),
                taker_sell_volume=volume * Decimal("0.45"),
            )
            session.add(bar)
            
            base_price = close_price
        
        session.commit()
    print("Market bars created successfully.")


def generate_sample_trades(db: Database, count: int = 50) -> None:
    """Generate sample trade data."""
    print(f"Generating {count} sample trades...")
    
    with db.get_session() as session:
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(count):
            trade_time = base_time + timedelta(hours=i * 3)
            order_id = f"order_{i:04d}"
            
            # Alternate sides
            side = "BUY" if i % 2 == 0 else "SELL"
            price = Decimal("50000.00") + Decimal(str(i))
            quantity = Decimal("0.0") + Decimal(str(round(abs(hash(str(i))) % 100) / 100 + 0.01))
            quote_quantity = price * quantity
            
            trade = Trade(
                order_id=order_id,
                trade_id=f"trade_{i:06d}",
                symbol="BTCUSDT",
                exchange="binance",
                side=side,
                price=price,
                quantity=quantity,
                quote_quantity=quote_quantity,
                commission=quote_quantity * Decimal("0.0005"),
                commission_asset="USDT",
                executed_at=trade_time,
                strategy_id=f"strat_{(i % 3) + 1:03d}",
            )
            session.add(trade)
        
        session.commit()
    print("Trades created successfully.")


def generate_sample_equity_curve(db: Database, days: int = 30) -> None:
    """Generate sample equity curve data."""
    print(f"Generating {days} days of equity curve data...")
    
    with db.get_session() as session:
        base_equity = Decimal("100000.00")
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days - i)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
            # Random daily return
            daily_return_pct = round((hash(str(i)) % 500 - 250) / 100, 2)
            net_change = base_equity * Decimal(str(daily_return_pct)) / 100
            
            closed_positions = abs(hash(str(i))) % 20 + 5
            winning_trades = int(closed_positions * (0.3 + (hash(str(i * 2)) % 40) / 100))
            losing_trades = closed_positions - winning_trades
            
            curve = EquityCurve(
                date=date,
                account="main",
                starting_equity=base_equity,
                ending_equity=base_equity + net_change,
                net_change=net_change,
                net_change_pct=daily_return_pct,
                deposits=Decimal("0"),
                withdrawals=Decimal("0") if i > 0 else Decimal("10000.00"),
                realized_pnl=net_change * Decimal("0.8"),
                unrealized_pnl=net_change * Decimal("0.2"),
                fees_paid=Decimal(str(round(abs(hash(str(i))) % 50, 2))),
                funding_fees=Decimal(str(round(abs(hash(str(i * 3))) % 10, 2))),
                open_positions=abs(hash(str(i))) % 5 + 1,
                closed_positions=closed_positions,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                max_drawdown=Decimal(str(round(abs(hash(str(i * 7))) % 500, 2))),
                max_drawdown_pct=round(abs(hash(str(i * 11))) % 200 / 100, 2),
            )
            session.add(curve)
            base_equity = base_equity + net_change
        
        session.commit()
    print("Equity curve created successfully.")


def generate_sample_strategy_performance(db: Database) -> None:
    """Generate sample strategy performance records."""
    print("Generating strategy performance records...")
    
    with db.get_session() as session:
        strategies = [
            ("RSI Mean Reversion", "strat_001"),
            ("MACD Cross", "strat_002"),
            ("Bollinger Breakout", "strat_003"),
        ]
        
        for strategy_name, strategy_id in strategies:
            # Monthly performance
            period_start = datetime.utcnow() - timedelta(days=30)
            period_end = datetime.utcnow()
            
            total_return = Decimal(str(round((hash(strategy_id) % 5000 - 2000) / 100, 2)))
            total_return_pct = round(float(total_return) / 1000 * 100, 2)
            
            total_trades = abs(hash(strategy_id)) % 200 + 50
            win_rate = round(0.3 + (hash(strategy_id * 2) % 50) / 100, 2)
            winning_trades = int(total_trades * win_rate)
            losing_trades = total_trades - winning_trades
            
            avg_win = Decimal(str(round(abs(hash(strategy_id * 3)) % 500 + 50, 2)))
            avg_loss = Decimal(str(round(abs(hash(strategy_id * 5)) % 200 + 20, 2)))
            
            perf = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                account="main",
                period_start=period_start,
                period_end=period_end,
                total_return=total_return,
                total_return_pct=total_return_pct,
                sharpe_ratio=round(0.5 + (hash(strategy_id * 7) % 300) / 100, 2),
                sortino_ratio=round(0.7 + (hash(strategy_id * 11) % 400) / 100, 2),
                max_drawdown=Decimal(str(round(abs(hash(strategy_id * 13)) % 1000, 2))),
                max_drawdown_pct=round(abs(hash(strategy_id * 17)) % 300 / 100, 2),
                calmar_ratio=round(1 + (hash(strategy_id * 19)) % 500 / 100, 2),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=round(float(avg_win) / float(avg_loss), 2) if avg_loss > 0 else 0,
                avg_trade_return=avg_win * Decimal(str(win_rate)) - avg_loss * Decimal(str(1 - win_rate)),
                daily_volatility=round(0.01 + (hash(strategy_id * 23)) % 200 / 10000, 4),
                annualized_volatility=round(0.15 + (hash(strategy_id * 29)) % 300 / 1000, 2),
            )
            session.add(perf)
        
        session.commit()
    print("Strategy performance created successfully.")


def generate_sample_parameters_log(db: Database) -> None:
    """Generate sample parameters log."""
    print("Generating parameters log...")
    
    with db.get_session() as session:
        strategies = [
            ("RSI Mean Reversion", "strat_001"),
            ("MACD Cross", "strat_002"),
        ]
        
        params = [
            ("rsi_period", "14", "21", "Optimization result"),
            ("rsi_overbought", "70", "75", "Reduced false signals"),
            ("rsi_oversold", "30", "25", "Catch more reversals"),
            ("position_size", "0.1", "0.15", "Increased conviction"),
        ]
        
        for strategy_name, strategy_id in strategies:
            for param_name, old_val, new_val, reason in params:
                log = ParametersLog(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    account="main",
                    parameter_name=param_name,
                    old_value=old_val,
                    new_value=new_val,
                    change_reason=reason,
                    change_type="optimization",
                    optimization_id=f"opt_{strategy_id}_{param_name}",
                    changed_at=datetime.utcnow() - timedelta(days=abs(hash(param_name)) % 30),
                    full_snapshot={
                        param_name: new_val for param_name, _, _, _ in params
                    },
                )
                session.add(log)
        
        session.commit()
    print("Parameters log created successfully.")


def generate_sample_funding_rates(db: Database, days: int = 30) -> None:
    """Generate sample funding rate history."""
    print(f"Generating {days * 3} funding rate records...")
    
    with db.get_session() as session:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        for symbol in symbols:
            base_rate = Decimal("0.0001")
            base_time = datetime.utcnow() - timedelta(days=days)
            
            for i in range(days * 3):  # 3 funding events per day (8h intervals)
                funding_time = base_time + timedelta(hours=i * 8)
                
                # Random rate variation
                rate_change = Decimal(str(round((hash(f"{symbol}{i}") % 100 - 50) / 100000, 6)))
                funding_rate = base_rate + rate_change
                
                rate = FundingRate(
                    symbol=symbol,
                    exchange="binance",
                    funding_rate=funding_rate,
                    funding_rate_pct=float(funding_rate) * 3 * 365 * 100,  # Annualized
                    mark_price=Decimal("50000.00") + Decimal(str(i * 10)),
                    index_price=Decimal("49999.00") + Decimal(str(i * 10)),
                    funding_time=funding_time,
                    next_funding_time=funding_time + timedelta(hours=8),
                    predicted_rate=funding_rate + Decimal(str(round((hash(f"{symbol}{i+1}") % 20 - 10) / 100000, 6))),
                )
                session.add(rate)
        
        session.commit()
    print("Funding rates created successfully.")


def generate_sample_order_logs(db: Database, count: int = 100) -> None:
    """Generate sample order logs."""
    print(f"Generating {count} sample order logs...")
    
    with db.get_session() as session:
        base_time = datetime.utcnow() - timedelta(days=7)
        statuses = ["NEW", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED"]
        
        for i in range(count):
            created_at = base_time + timedelta(hours=i * 2)
            status = statuses[i % len(statuses)]
            side = "BUY" if i % 2 == 0 else "SELL"
            
            is_filled = status == "FILLED"
            
            order = OrderLog(
                order_id=f"order_{i:06d}",
                client_order_id=f"cli_{i:06d}",
                exchange_order_id=f"ex_{i:06d}",
                symbol="BTCUSDT",
                exchange="binance",
                side=side,
                order_type="LIMIT" if i % 3 != 0 else "MARKET",
                status=status,
                position_side="LONG" if side == "BUY" else "SHORT",
                requested_price=Decimal("50000.00") + Decimal(str(i)),
                requested_quantity=Decimal(str(round(abs(hash(str(i))) % 100) / 100 + 0.01)),
                filled_price=Decimal("50000.00") + Decimal(str(i)) if is_filled else None,
                filled_quantity=Decimal("0.5") if is_filled else None,
                avg_fill_price=Decimal("50000.00") + Decimal(str(i)) if is_filled else None,
                commission=Decimal("2.5") if is_filled else Decimal("0"),
                commission_asset="USDT",
                created_at=created_at,
                updated_at=created_at + timedelta(minutes=1),
                filled_at=created_at + timedelta(minutes=2) if is_filled else None,
                strategy_id=f"strat_{(i % 3) + 1:03d}",
                strategy_name=["RSI Mean Reversion", "MACD Cross", "Bollinger Breakout"][i % 3],
                retry_count=0 if status != "REJECTED" else abs(hash(str(i))) % 3,
                error_message="Insufficient balance" if status == "REJECTED" else None,
                tags=["test"] if i % 10 == 0 else None,
            )
            session.add(order)
        
        session.commit()
    print("Order logs created successfully.")


def seed_database(db: Database) -> None:
    """Seed the database with sample data."""
    print("\n" + "=" * 60)
    print("SEEDING DATABASE WITH SAMPLE DATA")
    print("=" * 60 + "\n")
    
    generate_sample_market_bars(db, count=100)
    generate_sample_trades(db, count=50)
    generate_sample_equity_curve(db, days=30)
    generate_sample_strategy_performance(db)
    generate_sample_parameters_log(db)
    generate_sample_funding_rates(db, days=30)
    generate_sample_order_logs(db, count=100)
    
    print("\n" + "=" * 60)
    print("DATABASE SEEDING COMPLETE")
    print("=" * 60 + "\n")


# =============================================================================
# REDIS INITIALIZATION
# =============================================================================


def test_redis_connection() -> bool:
    """Test Redis connection and print status."""
    print("\nTesting Redis connection...")
    
    try:
        client = get_redis_client()
        if client.ping():
            print("✓ Redis connection successful")
            
            # Test basic operations
            client.set("test_key", "test_value", ex=10)
            result = client.get("test_key", decode=False)
            client.delete("test_key")
            
            if result == "test_value":
                print("✓ Redis read/write operations working")
                return True
            else:
                print("✗ Redis read/write operations failed")
                return False
        else:
            print("✗ Redis ping failed")
            return False
    except Exception as e:
        print(f"✗ Redis connection error: {e}")
        print("  Note: Redis is optional for basic database operations")
        print("  The database will work without Redis, but caching won't be available")
        return False


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Mimia Quant Trading System database"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all existing tables before creating",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the database with sample data",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/mimia_quant.db",
        help="Path to database file (default: data/mimia_quant.db)",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--test-redis",
        action="store_true",
        help="Test Redis connection and exit",
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    if not os.path.isabs(args.path):
        # Make relative to project root
        project_root = Path(__file__).parent.parent
        db_path = project_root / args.path
    else:
        db_path = Path(args.path)
    
    # Create data directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MIMIA QUANT TRADING SYSTEM - DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"\nDatabase path: {db_path}")
    print(f"Redis: {args.redis_host}:{args.redis_port}")
    
    # Test Redis first if requested
    if args.test_redis:
        # Create a temporary Redis client with custom host/port
        import core.redis_client as rc
        rc._instance = None  # Reset singleton
        rc._instance = RedisClient(host=args.redis_host, port=args.redis_port)
        test_redis_connection()
        rc._instance = None  # Reset
        return
    
    # Initialize database
    print("\n" + "-" * 60)
    print("Initializing database...")
    
    db = Database(db_path=str(db_path), echo=False)
    
    if args.drop:
        print("Dropping existing tables...")
        db.drop_tables()
        print("Tables dropped.")
    
    print("Creating tables...")
    db.create_tables()
    print("Tables created successfully.")
    
    # Get table info
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"\nTables created: {', '.join(tables)}")
    
    # Count records in each table
    with db.get_session() as session:
        print("\nTable record counts:")
        for table in tables:
            count = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
            print(f"  {table}: {count}")
    
    # Test Redis connection
    test_redis_connection()
    
    # Seed data if requested
    if args.seed:
        seed_database(db)
        
        # Print final counts
        with db.get_session() as session:
            print("\nFinal table record counts:")
            for table in tables:
                count = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                print(f"  {table}: {count}")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
