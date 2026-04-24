"""
Mimia Quant Trading System - Database Schema
SQLAlchemy models for SQLite database
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

Base = declarative_base()


class MarketBars(Base):
    """Market OHLCV bar data"""
    __tablename__ = 'market_bars'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default='1m')
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    quote_volume = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    taker_buy_volume = Column(Float, default=0.0)
    taker_buy_quote_volume = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uix_symbol_timeframe_timestamp'),
        Index('idx_market_bars_symbol_time', 'symbol', 'timeframe', 'timestamp'),
    )


class Trades(Base):
    """Executed trades record"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY or SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    quote_quantity = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(20), default='USDT')
    executed_at = Column(DateTime, nullable=False, index=True)
    order_id = Column(String(50), index=True)
    is_maker = Column(Boolean, default=False)
    is_best_match = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_trades_symbol_executed', 'symbol', 'executed_at'),
    )


class EquityCurve(Base):
    """Portfolio equity curve tracking"""
    __tablename__ = 'equity_curve'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, default=0.0)
    total_value = Column(Float, nullable=False)
    daily_pnl = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)
    cumulative_return = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    strategy_name = Column(String(50), index=True)
    session_id = Column(String(50), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_equity_strategy_session', 'strategy_name', 'session_id', 'timestamp'),
    )


class StrategyPerformance(Base):
    """Strategy performance metrics"""
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)
    risk_reward_ratio = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('strategy_name', 'session_id', 'timestamp', name='uix_strategy_session_time'),
    )


class ParametersLog(Base):
    """Strategy parameters logging for reproducibility"""
    __tablename__ = 'parameters_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), index=True)
    parameter_name = Column(String(100), nullable=False)
    parameter_value = Column(Text, nullable=False)
    parameter_type = Column(String(20), default='string')  # string, int, float, bool, json
    timestamp = Column(DateTime, nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_params_strategy_session', 'strategy_name', 'session_id'),
    )


class FundingRates(Base):
    """Cryptocurrency funding rates history"""
    __tablename__ = 'funding_rates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    funding_rate = Column(Float, nullable=False)
    funding_rate_predicted = Column(Float)
    mark_price = Column(Float)
    index_price = Column(Float)
    next_funding_time = Column(DateTime)
    interval_hours = Column(Integer, default=8)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uix_funding_symbol_timestamp'),
        Index('idx_funding_symbol_time', 'symbol', 'timestamp'),
    )


class OrderLog(Base):
    """Order execution log for audit trail"""
    __tablename__ = 'order_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    client_order_id = Column(String(50), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY or SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP, etc.
    status = Column(String(20), nullable=False, index=True)  # NEW, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED
    price = Column(Float)
    stop_price = Column(Float)
    original_quantity = Column(Float, nullable=False)
    executed_quantity = Column(Float, default=0.0)
    remaining_quantity = Column(Float)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(20), default='USDT')
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime)
    filled_at = Column(DateTime)
    strategy_name = Column(String(50), index=True)
    session_id = Column(String(50), index=True)
    notes = Column(Text)
    error_message = Column(Text)
    created_at_record = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_order_symbol_status', 'symbol', 'status'),
        Index('idx_order_strategy_session', 'strategy_name', 'session_id'),
    )


class Database:
    """Database connection and session management"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get('DATABASE_PATH', 'data/mimia_quant.db')
        
        if not db_path.startswith('/'):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, db_path)
        
        self.db_path = db_path
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            echo=False,
            connect_args={'check_same_thread': False},
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
    
    def create_all(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
    
    def drop_all(self):
        """Drop all tables"""
        Base.metadata.drop_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database engine"""
        self.engine.dispose()
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables"""
        self.drop_all()
        self.create_all()


def get_database(db_path: str = None) -> Database:
    """Factory function to get Database instance"""
    return Database(db_path)


def init_database(db_path: str = None) -> Database:
    """Initialize database and create all tables"""
    db = Database(db_path)
    db.create_all()
    return db
