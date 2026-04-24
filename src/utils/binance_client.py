"""
Binance REST API Client for Mimia Quant Trading System.
Uses python-binance library with testnet support.
"""

import os
from typing import Optional, Dict, Any, List
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import (
    OrderSide,
    OrderType,
    PositionSide,
    TimeInForce,
    FutureOrderType
)


class BinanceRESTClient:
    """
    Binance Futures REST API client with testnet support.
    Provides methods for account info, order management, and market data.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        testnet_api_key: Optional[str] = None,
        testnet_api_secret: Optional[str] = None
    ):
        """
        Initialize Binance REST client.
        
        Args:
            api_key: Binance API key (or env BINANCE_API_KEY)
            api_secret: Binance API secret (or env BINANCE_API_SECRET)
            testnet: Use testnet (default True)
            testnet_api_key: Testnet API key (or env BINANCE_TESTNET_API_KEY)
            testnet_api_secret: Testnet API secret (or env BINANCE_TESTNET_API_SECRET)
        """
        self.testnet = testnet
        
        # Get credentials from env if not provided
        if testnet:
            self.api_key = testnet_api_key or os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = testnet_api_secret or os.getenv("BINANCE_TESTNET_API_SECRET", "")
            self.base_url = "https://testnet.binancefuture.com"
            self.client = Client(self.api_key, self.api_secret, testnet=True)
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
            self.base_url = "https://fapi.binance.com"
            self.client = Client(self.api_key, self.api_secret)
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including balances and positions."""
        try:
            return self.client.futures_account()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get account info: {e}")
    
    def get_balance(self) -> List[Dict[str, Any]]:
        """Get futures account balance."""
        try:
            return self.client.futures_account_balance()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get balance: {e}")
    
    def get_position_info(self) -> List[Dict[str, Any]]:
        """Get current positions information."""
        try:
            return self.client.futures_position_information()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get position info: {e}")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        try:
            return self.client.futures_exchange_info()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get exchange info: {e}")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific trading pair."""
        try:
            exchange_info = self.get_exchange_info()
            for sym in exchange_info.get("symbols", []):
                if sym.get("symbol") == symbol.upper():
                    return sym
            return None
        except Exception as e:
            raise Exception(f"Failed to get symbol info: {e}")
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of klines to return (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
        
        Returns:
            List of klines, each containing [open_time, open, high, low, close, volume, close_time, ...]
        """
        try:
            return self.client.futures_klines(
                symbol=symbol.upper(),
                interval=interval,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to get klines: {e}")
    
    def get_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get order book depth for a symbol."""
        try:
            return self.client.futures_order_book(symbol=symbol.upper(), limit=limit)
        except BinanceAPIException as e:
            raise Exception(f"Failed to get order book: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        try:
            return self.client.futures_recent_trades(symbol=symbol.upper(), limit=limit)
        except BinanceAPIException as e:
            raise Exception(f"Failed to get recent trades: {e}")
    
    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        """Get current mark price for a symbol."""
        try:
            return self.client.futures_mark_price(symbol=symbol.upper())
        except BinanceAPIException as e:
            raise Exception(f"Failed to get mark price: {e}")
    
    def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get funding rate history for a symbol."""
        try:
            return self.client.futures_funding_rate(symbol=symbol.upper(), limit=limit)
        except BinanceAPIException as e:
            raise Exception(f"Failed to get funding rate: {e}")
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders, optionally filtered by symbol."""
        try:
            if symbol:
                return self.client.futures_get_open_orders(symbol=symbol.upper())
            return self.client.futures_get_open_orders()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get open orders: {e}")
    
    def get_all_orders(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all orders for a symbol (including history)."""
        try:
            return self.client.futures_get_all_orders(
                symbol=symbol.upper(),
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to get all orders: {e}")
    
    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get details of a specific order."""
        try:
            return self.client.futures_get_order(
                symbol=symbol.upper(),
                orderId=order_id,
                origClientOrderId=orig_client_order_id
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to get order: {e}")
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = TimeInForce.GTC,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
        new_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new futures order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            order_type: Order type ('LIMIT', 'MARKET', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET')
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            stop_price: Stop price (required for STOP and TAKE_PROFIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            reduce_only: If true, order only reduces position
            position_side: Position side ('LONG', 'SHORT', or None for both)
            new_client_order_id: Custom order ID
        
        Returns:
            Order response from Binance
        """
        try:
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity,
                "timeInForce": time_in_force,
                "reduceOnly": reduce_only
            }
            
            if price is not None:
                params["price"] = price
            
            if stop_price is not None:
                params["stopPrice"] = stop_price
            
            if position_side is not None:
                params["positionSide"] = position_side.upper()
            
            if new_client_order_id is not None:
                params["newClientOrderId"] = new_client_order_id
            
            return self.client.futures_create_order(**params)
        except BinanceAPIException as e:
            raise Exception(f"Failed to create order: {e}")
    
    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a market order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity,
            reduce_only=reduce_only,
            position_side=position_side
        )
    
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = TimeInForce.GTC,
        reduce_only: bool = False,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a limit order."""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            position_side=position_side
        )
    
    def create_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a stop order."""
        order_type = "STOP" if price else "STOP_MARKET"
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            position_side=position_side
        )
    
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an open order."""
        try:
            return self.client.futures_cancel_order(
                symbol=symbol.upper(),
                orderId=order_id,
                origClientOrderId=orig_client_order_id
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to cancel order: {e}")
    
    def cancel_all_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        try:
            return self.client.futures_cancel_all_open_orders(symbol=symbol.upper())
        except BinanceAPIException as e:
            raise Exception(f"Failed to cancel all orders: {e}")
    
    def cancel_orders(
        self,
        symbol: str,
        order_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Cancel multiple orders by their IDs."""
        try:
            return self.client.futures_cancel_orders(
                symbol=symbol.upper(),
                orderIdList=order_ids
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to cancel orders: {e}")
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol."""
        try:
            return self.client.futures_change_leverage(
                symbol=symbol.upper(),
                leverage=leverage
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to set leverage: {e}")
    
    def set_margin_type(
        self,
        symbol: str,
        margin_type: str,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Set margin type for a symbol.
        
        Args:
            symbol: Trading pair
            margin_type: 'ISOLATED' or 'CROSSED'
            reduce_only: Whether to enable reduce only
        """
        try:
            return self.client.futures_change_margin_type(
                symbol=symbol.upper(),
                marginType=margin_type.upper(),
                reduceOnly=reduce_only
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to set margin type: {e}")
    
    def get_leverage_bracket(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get leverage bracket for a symbol or all symbols."""
        try:
            if symbol:
                return self.client.futures_leverage_bracket(symbol=symbol.upper())
            return self.client.futures_leverage_bracket()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get leverage bracket: {e}")
    
    def get_position_mode(self) -> Dict[str, Any]:
        """Get current position mode (hedge mode or one-way)."""
        try:
            return self.client.futures_get_position_mode()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get position mode: {e}")
    
    def set_position_mode(self, hedge_mode: bool) -> Dict[str, Any]:
        """Set position mode (hedge mode or one-way)."""
        try:
            return self.client.futures_change_position_mode(dualCompassPosition=hedge_mode)
        except BinanceAPIException as e:
            raise Exception(f"Failed to set position mode: {e}")
    
    def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get income history (trading fee, realized PNL, funding fee, etc.)."""
        try:
            return self.client.futures_income_history(
                symbol=symbol.upper() if symbol else None,
                incomeType=income_type,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to get income history: {e}")
    
    def get_notional_and_leverage(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get notional and leverage bracketed info."""
        try:
            if symbol:
                return self.client.futures_notional_and_leverage(symbol=symbol.upper())
            return self.client.futures_notional_and_leverage()
        except BinanceAPIException as e:
            raise Exception(f"Failed to get notional and leverage: {e}")
    
    def get_account_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get account trade history for a symbol."""
        try:
            return self.client.futures_account_trades(
                symbol=symbol.upper(),
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            raise Exception(f"Failed to get account trades: {e}")
    
    def get_server_time(self) -> Dict[str, Any]:
        """Get server time."""
        try:
            return self.client.get_server_time()
        except Exception as e:
            raise Exception(f"Failed to get server time: {e}")
    
    def ping(self) -> Dict[str, Any]:
        """Ping the Binance API to check connectivity."""
        try:
            return self.client.ping()
        except Exception as e:
            raise Exception(f"Failed to ping: {e}")


# Convenience function for quick client creation
def create_binance_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = True
) -> BinanceRESTClient:
    """Create and return a Binance REST client."""
    return BinanceRESTClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
