#!/usr/bin/env python3
"""
Mimia Quant Trading Bot Orchestrator

Main entry point for the Mimia Quant trading system. Coordinates all components
including strategies, execution engine, risk management, and monitoring.

Usage:
    python scripts/run_bot.py [--config CONFIG_PATH] [--mode MODE] [--session SESSION_ID]

Modes:
    live    - Live trading with real money
    paper   - Paper trading (simulated execution)
    backtest - Historical backtesting

Environment Variables:
    ENVIRONMENT      - Set to 'production' for live trading
    REDIS_HOST      - Redis host for caching
    REDIS_PORT      - Redis port
    BINANCE_API_KEY - Exchange API key
    BINANCE_API_SECRET - Exchange API secret
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config, get_config
from src.core.logging import setup_logging, get_logger, TradingLogger
from src.core.database import Database, init_database
from src.core.redis_client import RedisClient, RedisManager
from src.core.constants import VERSION, SYSTEM_NAME

# Import strategies
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid import GridStrategy
from src.strategies.breakout import BreakoutStrategy

# Import execution components
from src.execution.execution_engine import ExecutionEngine, Order, OrderSide, OrderType, PositionSide
from src.execution.risk_manager import RiskManager
from src.execution.position_sizer import PositionSizer

# Import monitoring
from src.monitoring.monitor import Monitor, MonitorConfig
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.telegram_notifier import TelegramNotifier


class TradingBotOrchestrator:
    """
    Main orchestrator for the Mimia Quant trading system.
    
    Coordinates:
    - Strategy execution
    - Order execution and tracking
    - Risk management
    - Performance monitoring
    - State persistence
    """
    
    def __init__(
        self,
        config: Config,
        mode: str = "paper",
        session_id: Optional[str] = None
    ):
        """
        Initialize the trading bot orchestrator.
        
        Args:
            config: Configuration object
            mode: Trading mode (live, paper, backtest)
            session_id: Optional session identifier
        """
        self.config = config
        self.mode = mode
        self.session_id = session_id or f"{mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Logging
        self.logger = get_logger(f"{__name__}.Orchestrator")
        self.logger.info(f"Initializing Mimia Quant Trading Bot - Mode: {mode}, Session: {self.session_id}")
        
        # State
        self._running = False
        self._strategies: Dict[str, Any] = {}
        self._strategy_configs: Dict[str, Dict] = {}
        self._positions: Dict[str, Dict] = {}
        self._orders: Dict[str, Order] = {}
        
        # Components (initialized in setup)
        self.database: Optional[Database] = None
        self.redis_client: Optional[RedisClient] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.monitor: Optional[Monitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.telegram_notifier: Optional[TelegramNotifier] = None
        
        # Trading state
        self._portfolio_value = 10000.0  # Default starting portfolio
        self._equity_curve: List[Dict] = []
        self._last_trade_time: Dict[str, datetime] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        """Register handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the trading bot."""
        self.logger.info("Shutting down trading bot...")
        self._running = False
        
        try:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop()
                self.logger.info("Monitor stopped")
            
            # Cancel open orders
            if self.execution_engine:
                for order_id, order in list(self.execution_engine.orders.items()):
                    if order.is_active:
                        self.logger.info(f"Cancelling order {order_id}")
                        await self.execution_engine.cancel_order(order_id)
            
            # Save final state
            await self._save_state()
            
            # Send shutdown notification
            if self.telegram_notifier:
                self.telegram_notifier.send_alert(
                    title="Trading Bot Shutdown",
                    message=f"Session {self.session_id} has been shut down at {datetime.utcnow().isoformat()}",
                    level="info"
                )
            
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # ==================== Setup Methods ====================
    
    def setup(self) -> None:
        """Set up all components of the trading system."""
        self.logger.info("Setting up trading system components...")
        
        # Initialize database
        self._setup_database()
        
        # Initialize Redis
        self._setup_redis()
        
        # Initialize risk management
        self._setup_risk_management()
        
        # Initialize execution engine
        self._setup_execution_engine()
        
        # Initialize strategies
        self._setup_strategies()
        
        # Initialize monitoring
        self._setup_monitoring()
        
        # Load persisted state
        self._load_state()
        
        self.logger.info("System setup complete")
    
    def _setup_database(self) -> None:
        """Initialize database connection and create tables."""
        self.logger.info("Initializing database...")
        try:
            db_path = os.environ.get('DATABASE_PATH', str(PROJECT_ROOT / 'data' / 'mimia_quant.db'))
            self.database = init_database(db_path)
            self.logger.info(f"Database initialized at {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _setup_redis(self) -> None:
        """Initialize Redis connection."""
        self.logger.info("Initializing Redis connection...")
        try:
            redis_config = self.config.redis_config
            self.redis_client = RedisManager.get_instance(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                db=redis_config.get('db', 0)
            )
            
            # Test connection
            if self.redis_client.ping():
                self.logger.info("Redis connection established")
            else:
                self.logger.warning("Redis ping failed, continuing without Redis cache")
                self.redis_client = None
                
        except Exception as e:
            self.logger.warning(f"Redis initialization failed: {e}, continuing without Redis")
            self.redis_client = None
    
    def _setup_risk_management(self) -> None:
        """Initialize risk management components."""
        self.logger.info("Initializing risk management...")
        
        risk_config = self.config.risk_config
        self.risk_manager = RiskManager(config=risk_config)
        self.position_sizer = PositionSizer(config=risk_config)
        
        self.logger.info("Risk management initialized")
    
    def _setup_execution_engine(self) -> None:
        """Initialize the execution engine."""
        self.logger.info("Initializing execution engine...")
        
        simulate = self.mode in ('paper', 'backtest')
        
        self.execution_engine = ExecutionEngine(
            broker_client=None,  # Would be real broker client in live mode
            risk_manager=self.risk_manager,
            position_sizer=self.position_sizer,
            redis_client=self.redis_client,
            simulate=simulate
        )
        
        # Register callbacks
        self.execution_engine.set_order_callback(self._on_order_update)
        self.execution_engine.set_fill_callback(self._on_fill)
        self.execution_engine.set_position_callback(self._on_position_update)
        
        self.logger.info(f"Execution engine initialized (simulate={simulate})")
    
    def _setup_strategies(self) -> None:
        """Initialize all configured strategies."""
        self.logger.info("Initializing strategies...")
        
        # Get strategy configurations
        strategies_config = self.config.get_all_strategies()
        
        strategy_classes = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'grid': GridStrategy,
            'breakout': BreakoutStrategy,
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            strategy_cfg = strategies_config.get(strategy_name, {})
            
            if not strategy_cfg.get('enabled', False):
                self.logger.info(f"Strategy '{strategy_name}' is disabled, skipping")
                continue
            
            try:
                strategy = strategy_class(
                    name=strategy_name,
                    config=strategy_cfg
                )
                self._strategies[strategy_name] = strategy
                self._strategy_configs[strategy_name] = strategy_cfg
                
                # Register with monitor
                if self.monitor:
                    self.monitor.register_strategy(strategy_name)
                
                self.logger.info(f"Strategy '{strategy_name}' initialized and enabled")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy '{strategy_name}': {e}")
        
        self.logger.info(f"Initialized {len(self._strategies)} strategies")
    
    def _setup_monitoring(self) -> None:
        """Initialize monitoring and alerting."""
        self.logger.info("Initializing monitoring...")
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            redis_client=self.redis_client,
            database=self.database
        )
        
        # Initialize Telegram notifier if configured
        telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            self.telegram_notifier = TelegramNotifier(
                bot_token=telegram_token,
                chat_id=telegram_chat_id
            )
            self.logger.info("Telegram notifier initialized")
        else:
            self.logger.info("Telegram not configured, skipping notification setup")
        
        # Initialize monitor
        monitor_config = MonitorConfig(
            check_interval_seconds=60,
            edge_check_interval_seconds=300,
            daily_report_hour_utc=8,  # 08:00 UTC for daily report
            auto_pause_on_critical=True,
            auto_reduce_size_on_decay=True,
            reduction_factor=0.5
        )
        
        self.monitor = Monitor(
            config=monitor_config,
            metrics_collector=self.metrics_collector,
            telegram_notifier=self.telegram_notifier
        )
        
        # Set up callbacks
        self.monitor.set_pause_callback(self._pause_strategy)
        self.monitor.set_reduce_size_callback(self._reduce_strategy_size)
        
        # Register strategies with monitor
        for strategy_name in self._strategies:
            self.monitor.register_strategy(strategy_name)
        
        # Start monitor
        self.monitor.start()
        
        self.logger.info("Monitoring initialized")
    
    # ==================== Strategy Control Callbacks ====================
    
    def _pause_strategy(self, strategy_name: str) -> None:
        """Pause a strategy (called by monitor on critical decay)."""
        self.logger.warning(f"Pausing strategy: {strategy_name}")
        if strategy_name in self._strategies:
            self._strategies[strategy_name].enabled = False
    
    def _reduce_strategy_size(self, strategy_name: str, factor: float) -> None:
        """Reduce position size for a strategy (called by monitor on decay)."""
        self.logger.info(f"Reducing position size for {strategy_name} by factor {factor}")
        if strategy_name in self._strategy_configs:
            config = self._strategy_configs[strategy_name]
            current_size = config.get('position_size_pct', 0.1)
            config['position_size_pct'] = current_size * factor
    
    # ==================== Execution Callbacks ====================
    
    def _on_order_update(self, order: Order) -> None:
        """Handle order updates."""
        self.logger.info(f"Order update: {order.order_id} - {order.status.value}")
        self._orders[order.order_id] = order
        
        # Record in metrics
        if self.metrics_collector:
            self.metrics_collector.record_order(order)
    
    def _on_fill(self, fill) -> None:
        """Handle trade fills."""
        self.logger.info(f"Trade filled: {fill.order_id} - {fill.side.value} {fill.quantity} @ {fill.price}")
        
        # Record trade in metrics
        if self.metrics_collector:
            self.metrics_collector.record_trade(fill)
    
    def _on_position_update(self, symbol: str, position) -> None:
        """Handle position updates."""
        self._positions[symbol] = position
        
        # Record in metrics
        if self.metrics_collector:
            self.metrics_collector.record_position(symbol, position)
    
    # ==================== State Management ====================
    
    async def _save_state(self) -> None:
        """Save current state to persistent storage."""
        try:
            state = {
                'session_id': self.session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': self._portfolio_value,
                'positions': {symbol: pos.to_dict() if hasattr(pos, 'to_dict') else str(pos) 
                             for symbol, pos in self._positions.items()},
                'active_strategies': list(self._strategies.keys()),
                'equity_curve': self._equity_curve[-100:]  # Keep last 100 entries
            }
            
            if self.redis_client:
                self.redis_client.save_strategy_state('orchestrator', self.session_id, state)
            
            self.logger.debug("State saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load persisted state from storage."""
        try:
            if self.redis_client:
                state = self.redis_client.get_strategy_state('orchestrator', self.session_id)
                if state:
                    self._portfolio_value = state.get('portfolio_value', self._portfolio_value)
                    self._equity_curve = state.get('equity_curve', [])
                    self.logger.info(f"Loaded state: portfolio={self._portfolio_value}")
                else:
                    self.logger.info("No previous state found, starting fresh")
            
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")
    
    # ==================== Main Trading Loop ====================
    
    async def run(self) -> None:
        """Run the main trading loop."""
        self.logger.info("Starting main trading loop...")
        self._running = True
        
        # Send startup notification
        if self.telegram_notifier:
            self.telegram_notifier.send_alert(
                title="Trading Bot Started",
                message=f"Session {self.session_id} started in {self.mode} mode at {datetime.utcnow().isoformat()}",
                level="info"
            )
        
        try:
            while self._running:
                await self._trading_iteration()
                await asyncio.sleep(1)  # Main loop interval
                
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.send_alert(
                    title="Trading Bot Error",
                    message=f"Error in session {self.session_id}: {str(e)}",
                    level="error"
                )
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self) -> None:
        """Execute one iteration of the trading loop."""
        loop_start = datetime.utcnow()
        
        async with self._lock:
            # 1. Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # 2. Run strategy analysis
            signals = await self._run_strategy_analysis()
            
            # 3. Process signals and generate orders
            await self._process_signals(signals)
            
            # 4. Update positions
            await self._update_positions()
            
            # 5. Check risk limits
            await self._check_risk_limits()
            
            # 6. Record metrics
            self._record_metrics()
            
            # 7. Periodic state save
            if (loop_start.minute % 5 == 0 and loop_start.second < 2):
                await self._save_state()
        
        # Log loop completion
        loop_duration = (datetime.utcnow() - loop_start).total_seconds()
        self.logger.debug(f"Trading iteration completed in {loop_duration:.3f}s")
    
    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio value and metrics."""
        total_equity = self._portfolio_value
        
        # Calculate positions value
        positions_value = 0.0
        for symbol, position in self._positions.items():
            if hasattr(position, 'unrealized_pnl'):
                positions_value += position.unrealized_pnl
        
        # Calculate daily PnL (simplified)
        daily_pnl = positions_value
        daily_return = (positions_value / self._portfolio_value) * 100 if self._portfolio_value > 0 else 0
        
        # Update equity curve
        self._equity_curve.append({
            'timestamp': datetime.utcnow().isoformat(),
            'equity': total_equity,
            'positions_value': positions_value,
            'daily_pnl': daily_pnl
        })
        
        # Keep equity curve bounded
        if len(self._equity_curve) > 1000:
            self._equity_curve = self._equity_curve[-500:]
        
        # Record with monitor
        if self.monitor:
            self.monitor.record_portfolio_metrics(
                total_equity=total_equity,
                cash=self._portfolio_value - positions_value,
                positions_value=positions_value,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                cumulative_return=((total_equity - 10000) / 10000) * 100,  # Assuming 10k starting
                drawdown=self._calculate_drawdown()
            )
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from equity curve."""
        if not self._equity_curve:
            return 0.0
        
        equity_values = [e['equity'] for e in self._equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]
        
        return peak - current if peak > current else 0.0
    
    async def _run_strategy_analysis(self) -> List[Dict]:
        """Run analysis on all enabled strategies."""
        signals = []
        
        for strategy_name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            
            try:
                # Check cooldown
                if strategy_name in self._last_trade_time:
                    cooldown = strategy.config.get('cooldown_period_seconds', 300)
                    if (datetime.utcnow() - self._last_trade_time[strategy_name]).total_seconds() < cooldown:
                        continue
                
                # Get symbols to trade (in real implementation, this would come from a scanner)
                symbols = self._get_trading_symbols()
                
                for symbol in symbols:
                    # Get market data (simplified - in real impl would fetch from exchange)
                    data = await self._get_market_data(symbol, strategy.config.get('timeframe', '1h'))
                    
                    if data is not None:
                        signal = strategy.analyze(symbol, data)
                        
                        if signal is not None:
                            signals.append({
                                'strategy': strategy_name,
                                'signal': signal,
                                'symbol': symbol
                            })
                            self.logger.info(f"Signal generated: {strategy_name} on {symbol}")
                            
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {e}")
        
        return signals
    
    async def _get_market_data(self, symbol: str, timeframe: str):
        """Get market data for a symbol. Simplified implementation."""
        # In a real implementation, this would fetch from exchange API
        # For now, return None (no data = no signals)
        return None
    
    def _get_trading_symbols(self) -> List[str]:
        """Get list of symbols to trade."""
        # In a real implementation, this would be dynamic based on market scanner
        return ['BTCUSDT', 'ETHUSDT']
    
    async def _process_signals(self, signals: List[Dict]) -> None:
        """Process trading signals and create orders."""
        for signal_data in signals:
            strategy_name = signal_data['strategy']
            signal = signal_data['signal']
            symbol = signal_data['symbol']
            
            try:
                # Calculate position size
                position_size = await self._calculate_position_size(
                    signal, strategy_name
                )
                
                if position_size <= 0:
                    continue
                
                # Create order
                order = Order(
                    symbol=symbol,
                    side=signal.side,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    strategy_name=strategy_name,
                    session_id=self.session_id,
                    signal_id=signal.id
                )
                
                # Submit order
                if self.execution_engine:
                    result = await self.execution_engine.submit_order(order)
                    
                    if result.success:
                        self._last_trade_time[strategy_name] = datetime.utcnow()
                        self.logger.info(f"Order submitted: {result.order.order_id}")
                        
            except Exception as e:
                self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def _calculate_position_size(self, signal, strategy_name: str) -> float:
        """Calculate position size based on signal and risk rules."""
        config = self._strategy_configs.get(strategy_name, {})
        base_size_pct = config.get('position_size_pct', 0.1)
        
        # Adjust based on signal strength
        adjusted_size = base_size_pct * signal.strength
        
        # Cap at max position size
        max_size = self.config.max_position_size
        adjusted_size = min(adjusted_size, max_size)
        
        # Convert to quantity (simplified)
        return adjusted_size * self._portfolio_value
    
    async def _update_positions(self) -> None:
        """Update current positions."""
        if not self.execution_engine:
            return
        
        # Sync with execution engine positions
        for symbol, position in self.execution_engine.positions.items():
            self._positions[symbol] = position
    
    async def _check_risk_limits(self) -> None:
        """Check if risk limits are being breached."""
        # Check daily loss limit
        daily_loss_pct = (self._calculate_drawdown() / self._portfolio_value) * 100
        
        if daily_loss_pct > self.config.max_daily_loss * 100:
            self.logger.critical(f"Daily loss limit breached: {daily_loss_pct:.2f}%")
            if self.telegram_notifier:
                self.telegram_notifier.send_alert(
                    title="Risk Limit Alert",
                    message=f"Daily loss {daily_loss_pct:.2f}% exceeds limit",
                    level="critical"
                )
            
            # Pause all strategies
            for strategy_name in self._strategies:
                self._strategies[strategy_name].enabled = False
    
    def _record_metrics(self) -> None:
        """Record current metrics."""
        if not self.metrics_collector:
            return
        
        # Record health metrics
        import psutil
        
        try:
            self.monitor.record_health_metrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                active_strategies=sum(1 for s in self._strategies.values() if s.enabled),
                open_positions=len(self._positions),
                redis_connected=self.redis_client.ping() if self.redis_client else False,
                database_connected=True
            )
        except ImportError:
            pass  # psutil not available


class DailyReportJob:
    """Scheduled job for generating daily reports at 08:00 UTC."""
    
    def __init__(self, orchestrator: TradingBotOrchestrator):
        self.orchestrator = orchestrator
        self.logger = get_logger(f"{__name__}.DailyReport")
    
    async def run(self) -> None:
        """Generate and send daily report."""
        self.logger.info("Generating daily report...")
        
        try:
            if self.orchestrator.monitor:
                report = self.orchestrator.monitor.generate_daily_report()
                self.logger.info(f"Daily report generated:\n{report}")
                
                if self.orchestrator.telegram_notifier:
                    self.orchestrator.telegram_notifier.send_alert(
                        title="Daily Trading Report",
                        message=report,
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")


# ==================== CLI Entry Point ====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mimia Quant Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_bot.py --mode paper
  python scripts/run_bot.py --mode live --session my_session_001
  python scripts/run_bot.py --config /path/to/config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--session', '-s',
        type=str,
        default=None,
        help='Session ID for this run'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize configuration
    config = get_config()
    if args.config:
        config = Config(config_path=args.config)
    config.load()
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_dir=PROJECT_ROOT / 'logs',
        log_to_file=True,
        log_to_console=True
    )
    
    logger.info(f"{SYSTEM_NAME} v{VERSION} - Trading Bot")
    logger.info(f"Mode: {args.mode}, Session: {args.session or 'auto'}")
    
    try:
        # Create and setup orchestrator
        orchestrator = TradingBotOrchestrator(
            config=config,
            mode=args.mode,
            session_id=args.session
        )
        orchestrator.setup()
        
        # Run the trading bot
        asyncio.run(orchestrator.run())
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Trading bot terminated")


if __name__ == '__main__':
    main()
