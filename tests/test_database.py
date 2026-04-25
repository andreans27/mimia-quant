"""
Tests for Mimia Quant Database Module
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.core.database import (
    Base,
    Database,
    MarketBars,
    Trades,
    EquityCurve,
    StrategyPerformance,
    ParametersLog,
    FundingRates,
    OrderLog,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database file"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db_engine(temp_db_path):
    """Create a test database engine"""
    engine = create_engine(f'sqlite:///{temp_db_path}')
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a test database session"""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_market_bar():
    """Create a sample market bar"""
    return {
        'symbol': 'BTCUSDT',
        'timeframe': '1m',
        'timestamp': datetime(2024, 1, 15, 10, 30, 0),
        'open': 50000.0,
        'high': 50100.0,
        'low': 49900.0,
        'close': 50050.0,
        'volume': 100.5,
        'quote_volume': 5025025.0,
        'trades_count': 1500,
    }


@pytest.fixture
def sample_trade():
    """Create a sample trade"""
    return {
        'trade_id': '12345',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'price': 50000.0,
        'quantity': 0.1,
        'quote_quantity': 5000.0,
        'commission': 2.5,
        'commission_asset': 'USDT',
        'executed_at': datetime(2024, 1, 15, 10, 30, 0),
        'order_id': 'order_001',
        'is_maker': False,
        'is_best_match': True,
    }


@pytest.fixture
def sample_equity():
    """Create a sample equity curve entry"""
    return {
        'timestamp': datetime(2024, 1, 15, 10, 30, 0),
        'equity': 10000.0,
        'cash': 5000.0,
        'positions_value': 5000.0,
        'total_value': 10000.0,
        'daily_pnl': 100.0,
        'daily_return': 0.01,
        'cumulative_return': 0.01,
        'drawdown': 0.0,
        'strategy_name': 'ma_cross',
        'session_id': 'session_001',
    }


@pytest.fixture
def sample_strategy_performance():
    """Create a sample strategy performance entry"""
    return {
        'strategy_name': 'ma_cross',
        'session_id': 'session_001',
        'timestamp': datetime(2024, 1, 15, 10, 30, 0),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'win_rate': 0.6,
        'total_pnl': 1000.0,
        'total_pnl_pct': 0.1,
        'avg_win': 50.0,
        'avg_loss': -25.0,
        'profit_factor': 2.0,
        'max_drawdown': 100.0,
        'max_drawdown_pct': 0.02,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'calmar_ratio': 1.0,
        'volatility': 0.15,
        'risk_reward_ratio': 2.0,
    }


@pytest.fixture
def sample_order_log():
    """Create a sample order log entry"""
    return {
        'order_id': 'order_001',
        'client_order_id': 'client_001',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'order_type': 'LIMIT',
        'status': 'FILLED',
        'price': 50000.0,
        'stop_price': None,
        'original_quantity': 0.1,
        'executed_quantity': 0.1,
        'remaining_quantity': 0.0,
        'commission': 2.5,
        'commission_asset': 'USDT',
        'created_at': datetime(2024, 1, 15, 10, 0, 0),
        'updated_at': datetime(2024, 1, 15, 10, 30, 0),
        'filled_at': datetime(2024, 1, 15, 10, 30, 0),
        'strategy_name': 'ma_cross',
        'session_id': 'session_001',
        'notes': None,
        'error_message': None,
    }


# ==================== MarketBars Tests ====================

class TestMarketBars:
    def test_create_market_bar(self, db_session, sample_market_bar):
        """Test creating a market bar"""
        bar = MarketBars(**sample_market_bar)
        db_session.add(bar)
        db_session.commit()
        
        assert bar.id is not None
        assert bar.symbol == 'BTCUSDT'
        assert bar.timeframe == '1m'
        assert bar.close == 50050.0
    
    def test_market_bar_unique_constraint(self, db_session, sample_market_bar):
        """Test unique constraint on symbol, timeframe, timestamp"""
        bar1 = MarketBars(**sample_market_bar)
        db_session.add(bar1)
        db_session.commit()
        
        bar2 = MarketBars(**sample_market_bar)
        db_session.add(bar2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_market_bar_query_by_symbol(self, db_session, sample_market_bar):
        """Test querying market bars by symbol"""
        bar = MarketBars(**sample_market_bar)
        db_session.add(bar)
        db_session.commit()
        
        results = db_session.query(MarketBars).filter(
            MarketBars.symbol == 'BTCUSDT'
        ).all()
        assert len(results) == 1
        assert results[0].symbol == 'BTCUSDT'
    
    def test_market_bar_query_by_timeframe(self, db_session, sample_market_bar):
        """Test querying market bars by timeframe"""
        bar = MarketBars(**sample_market_bar)
        db_session.add(bar)
        db_session.commit()
        
        results = db_session.query(MarketBars).filter(
            MarketBars.timeframe == '1m'
        ).all()
        assert len(results) == 1
        
        results = db_session.query(MarketBars).filter(
            MarketBars.timeframe == '5m'
        ).all()
        assert len(results) == 0


# ==================== Trades Tests ====================

class TestTrades:
    def test_create_trade(self, db_session, sample_trade):
        """Test creating a trade"""
        trade = Trades(**sample_trade)
        db_session.add(trade)
        db_session.commit()
        
        assert trade.id is not None
        assert trade.trade_id == '12345'
        assert trade.side == 'BUY'
        assert trade.price == 50000.0
    
    def test_trade_unique_trade_id(self, db_session, sample_trade):
        """Test unique constraint on trade_id"""
        trade1 = Trades(**sample_trade)
        db_session.add(trade1)
        db_session.commit()
        
        trade2 = Trades(**sample_trade)
        db_session.add(trade2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_trade_query_by_symbol(self, db_session, sample_trade):
        """Test querying trades by symbol"""
        trade = Trades(**sample_trade)
        db_session.add(trade)
        db_session.commit()
        
        results = db_session.query(Trades).filter(
            Trades.symbol == 'BTCUSDT'
        ).all()
        assert len(results) == 1
    
    def test_trade_query_by_side(self, db_session, sample_trade):
        """Test querying trades by side"""
        trade = Trades(**sample_trade)
        db_session.add(trade)
        db_session.commit()
        
        results = db_session.query(Trades).filter(
            Trades.side == 'BUY'
        ).all()
        assert len(results) == 1
        
        results = db_session.query(Trades).filter(
            Trades.side == 'SELL'
        ).all()
        assert len(results) == 0


# ==================== EquityCurve Tests ====================

class TestEquityCurve:
    def test_create_equity_entry(self, db_session, sample_equity):
        """Test creating an equity curve entry"""
        equity = EquityCurve(**sample_equity)
        db_session.add(equity)
        db_session.commit()
        
        assert equity.id is not None
        assert equity.equity == 10000.0
        assert equity.strategy_name == 'ma_cross'
    
    def test_equity_query_by_strategy(self, db_session, sample_equity):
        """Test querying equity by strategy name"""
        equity = EquityCurve(**sample_equity)
        db_session.add(equity)
        db_session.commit()
        
        results = db_session.query(EquityCurve).filter(
            EquityCurve.strategy_name == 'ma_cross'
        ).all()
        assert len(results) == 1
        
        results = db_session.query(EquityCurve).filter(
            EquityCurve.strategy_name == 'rsi'
        ).all()
        assert len(results) == 0
    
    def test_equity_query_by_session(self, db_session, sample_equity):
        """Test querying equity by session ID"""
        equity = EquityCurve(**sample_equity)
        db_session.add(equity)
        db_session.commit()
        
        results = db_session.query(EquityCurve).filter(
            EquityCurve.session_id == 'session_001'
        ).all()
        assert len(results) == 1


# ==================== StrategyPerformance Tests ====================

class TestStrategyPerformance:
    def test_create_strategy_performance(self, db_session, sample_strategy_performance):
        """Test creating a strategy performance entry"""
        perf = StrategyPerformance(**sample_strategy_performance)
        db_session.add(perf)
        db_session.commit()
        
        assert perf.id is not None
        assert perf.strategy_name == 'ma_cross'
        assert perf.win_rate == 0.6
        assert perf.sharpe_ratio == 1.5
    
    def test_strategy_performance_unique_constraint(self, db_session, sample_strategy_performance):
        """Test unique constraint on strategy_name, session_id, timestamp"""
        perf1 = StrategyPerformance(**sample_strategy_performance)
        db_session.add(perf1)
        db_session.commit()
        
        perf2 = StrategyPerformance(**sample_strategy_performance)
        db_session.add(perf2)
        with pytest.raises(IntegrityError):
            db_session.commit()


# ==================== ParametersLog Tests ====================

class TestParametersLog:
    def test_create_parameter_log(self, db_session):
        """Test creating a parameter log entry"""
        param = ParametersLog(
            strategy_name='ma_cross',
            session_id='session_001',
            parameter_name='fast_ma_period',
            parameter_value='10',
            parameter_type='int',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            description='Fast moving average period'
        )
        db_session.add(param)
        db_session.commit()
        
        assert param.id is not None
        assert param.parameter_name == 'fast_ma_period'
        assert param.parameter_value == '10'
    
    def test_parameter_log_query_by_strategy(self, db_session):
        """Test querying parameters by strategy"""
        param = ParametersLog(
            strategy_name='ma_cross',
            session_id='session_001',
            parameter_name='fast_ma_period',
            parameter_value='10',
            parameter_type='int',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        db_session.add(param)
        db_session.commit()
        
        results = db_session.query(ParametersLog).filter(
            ParametersLog.strategy_name == 'ma_cross'
        ).all()
        assert len(results) == 1


# ==================== FundingRates Tests ====================

class TestFundingRates:
    def test_create_funding_rate(self, db_session):
        """Test creating a funding rate entry"""
        funding = FundingRates(
            symbol='BTCUSDT',
            timestamp=datetime(2024, 1, 15, 8, 0, 0),
            funding_rate=0.0001,
            funding_rate_predicted=0.00015,
            mark_price=50000.0,
            index_price=49999.0,
            next_funding_time=datetime(2024, 1, 15, 16, 0, 0),
            interval_hours=8,
        )
        db_session.add(funding)
        db_session.commit()
        
        assert funding.id is not None
        assert funding.symbol == 'BTCUSDT'
        assert funding.funding_rate == 0.0001
    
    def test_funding_rate_unique_constraint(self, db_session):
        """Test unique constraint on symbol and timestamp"""
        funding1 = FundingRates(
            symbol='BTCUSDT',
            timestamp=datetime(2024, 1, 15, 8, 0, 0),
            funding_rate=0.0001,
        )
        db_session.add(funding1)
        db_session.commit()
        
        funding2 = FundingRates(
            symbol='BTCUSDT',
            timestamp=datetime(2024, 1, 15, 8, 0, 0),
            funding_rate=0.0002,
        )
        db_session.add(funding2)
        with pytest.raises(IntegrityError):
            db_session.commit()


# ==================== OrderLog Tests ====================

class TestOrderLog:
    def test_create_order_log(self, db_session, sample_order_log):
        """Test creating an order log entry"""
        order = OrderLog(**sample_order_log)
        db_session.add(order)
        db_session.commit()
        
        assert order.id is not None
        assert order.order_id == 'order_001'
        assert order.status == 'FILLED'
        assert order.executed_quantity == 0.1
    
    def test_order_log_unique_order_id(self, db_session, sample_order_log):
        """Test unique constraint on order_id"""
        order1 = OrderLog(**sample_order_log)
        db_session.add(order1)
        db_session.commit()
        
        order2 = OrderLog(**sample_order_log)
        db_session.add(order2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_order_log_query_by_status(self, db_session, sample_order_log):
        """Test querying orders by status"""
        order = OrderLog(**sample_order_log)
        db_session.add(order)
        db_session.commit()
        
        results = db_session.query(OrderLog).filter(
            OrderLog.status == 'FILLED'
        ).all()
        assert len(results) == 1
        
        results = db_session.query(OrderLog).filter(
            OrderLog.status == 'PENDING'
        ).all()
        assert len(results) == 0
    
    def test_order_log_query_by_symbol_and_status(self, db_session, sample_order_log):
        """Test querying orders by symbol and status"""
        order = OrderLog(**sample_order_log)
        db_session.add(order)
        db_session.commit()
        
        results = db_session.query(OrderLog).filter(
            OrderLog.symbol == 'BTCUSDT',
            OrderLog.status == 'FILLED'
        ).all()
        assert len(results) == 1


# ==================== Database Class Tests ====================

class TestDatabase:
    def test_database_create_all(self, temp_db_path):
        """Test Database.create_all creates all tables"""
        db = Database(temp_db_path)
        db.create_all()
        
        # Verify tables exist by checking if we can query them
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()
        
        expected_tables = [
            'market_bars', 'trades', 'equity_curve', 
            'strategy_performance', 'parameters_log', 
            'funding_rates', 'order_log'
        ]
        for table in expected_tables:
            assert table in table_names
        
        db.close()
    
    def test_database_drop_all(self, temp_db_path):
        """Test Database.drop_all removes all tables"""
        db = Database(temp_db_path)
        db.create_all()
        db.drop_all()
        
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()
        assert len(table_names) == 0
        
        db.close()
    
    def test_database_get_session(self, temp_db_path):
        """Test Database.get_session returns a valid session"""
        db = Database(temp_db_path)
        db.create_all()
        
        session = db.get_session()
        assert session is not None
        session.close()
        
        db.close()
    
    def test_database_reset_database(self, temp_db_path):
        """Test Database.reset_database drops and recreates tables"""
        db = Database(temp_db_path)
        db.create_all()
        
        # Add some data
        session = db.get_session()
        bar = MarketBars(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        session.add(bar)
        session.commit()
        session.close()
        
        # Reset database
        db.reset_database()
        
        # Verify data is gone
        session = db.get_session()
        count = session.query(MarketBars).count()
        assert count == 0
        session.close()
        
        db.close()


# ==================== Integration Tests ====================

class TestDatabaseIntegration:
    def test_full_trading_workflow(self, db_session):
        """Test a full trading workflow with all tables"""
        # 1. Log parameters
        param = ParametersLog(
            strategy_name='ma_cross',
            session_id='session_001',
            parameter_name='fast_ma_period',
            parameter_value='10',
            parameter_type='int',
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        )
        db_session.add(param)
        
        # 2. Record market bar
        bar = MarketBars(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        db_session.add(bar)
        
        # 3. Create order
        order = OrderLog(
            order_id='order_001',
            symbol='BTCUSDT',
            side='BUY',
            order_type='MARKET',
            status='FILLED',
            original_quantity=0.1,
            executed_quantity=0.1,
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            filled_at=datetime(2024, 1, 15, 10, 30, 0),
            strategy_name='ma_cross',
            session_id='session_001',
        )
        db_session.add(order)
        
        # 4. Record trade
        trade = Trades(
            trade_id='trade_001',
            symbol='BTCUSDT',
            side='BUY',
            price=50050.0,
            quantity=0.1,
            quote_quantity=5005.0,
            executed_at=datetime(2024, 1, 15, 10, 30, 0),
            order_id='order_001',
        )
        db_session.add(trade)
        
        # 5. Update equity
        equity = EquityCurve(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            equity=10050.0,
            cash=5000.0,
            positions_value=5050.0,
            total_value=10050.0,
            strategy_name='ma_cross',
            session_id='session_001',
        )
        db_session.add(equity)
        
        # 6. Record strategy performance
        perf = StrategyPerformance(
            strategy_name='ma_cross',
            session_id='session_001',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=50.0,
        )
        db_session.add(perf)
        
        db_session.commit()
        
        # Verify all records exist
        assert db_session.query(ParametersLog).count() == 1
        assert db_session.query(MarketBars).count() == 1
        assert db_session.query(OrderLog).count() == 1
        assert db_session.query(Trades).count() == 1
        assert db_session.query(EquityCurve).count() == 1
        assert db_session.query(StrategyPerformance).count() == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
