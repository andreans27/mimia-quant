"""
Binance REST API Client for Mimia Quant Trading System.
Uses official binance-connector-python library (binance-sdk-derivatives-trading-usds-futures).
"""

import os
import sys
from typing import Optional, Dict, Any, List, Union

# Add the venv site-packages to path if needed
sys.path.insert(0, '/root/.hermes/hermes-agent/venv/lib/python3.11/site-packages')

from binance_sdk_derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
from binance_common.configuration import ConfigurationRestAPI

# Order constants
ORDER_SIDE_BUY = "BUY"
ORDER_SIDE_SELL = "SELL"
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_STOP = "STOP"
ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
POSITION_SIDE_LONG = "LONG"
POSITION_SIDE_SHORT = "SHORT"
TIME_IN_FORCE_GTC = "GTC"
TIME_IN_FORCE_IOC = "IOC"
TIME_IN_FORCE_FOK = "FOK"


class BinanceRESTClient:
    """
    Binance Futures REST API client using official binance-connector-python.
    Supports both testnet and production environments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize Binance REST client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (default True)
        """
        self.testnet = testnet

        # Get credentials from env if not provided
        if testnet:
            self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET", "")
            base_url = DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
            base_url = DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL

        # Create configuration and client
        config = ConfigurationRestAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_path=base_url,
        )
        self._client = DerivativesTradingUsdsFutures(config_rest_api=config)
        self._base_url = base_url

    def _call(self, method: str, **kwargs) -> Any:
        """
        Call a REST API method and return the data.

        Args:
            method: Method name on rest_api (e.g., 'exchange_information')
            **kwargs: Arguments to pass to the method

        Returns:
            The data from the API response
        """
        api = getattr(self._client.rest_api, method)
        response = api(**kwargs)
        return response.data()

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including balances and positions."""
        try:
            # Try v3 first (more recent)
            data = self._call('account_information_v3')
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get account info: {e}")

    def get_balance(self) -> List[Dict[str, Any]]:
        """Get futures account balance."""
        try:
            data = self._call('futures_account_balance_v3')
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get balance: {e}")

    def get_position_info(self) -> List[Dict[str, Any]]:
        """Get current positions information."""
        try:
            data = self._call('account_information_v3')
            if hasattr(data, 'positions'):
                return [self._pydantic_to_dict(p) if hasattr(p, 'model_dump') else p for p in data.positions]
            return []
        except Exception as e:
            raise Exception(f"Failed to get position info: {e}")

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        try:
            data = self._call('exchange_information')
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get exchange info: {e}")

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific trading pair."""
        try:
            exchange_info = self.get_exchange_info()
            symbols = exchange_info.get('symbols', [])
            for sym in symbols:
                if sym.get('symbol') == symbol.upper():
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
        """
        try:
            data = self._call(
                'kline_candlestick_data',
                symbol=symbol.upper(),
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            # Convert to list of lists format for backward compatibility
            if isinstance(data, list) and data:
                first = data[0]
                if hasattr(first, 'model_dump'):
                    return [[
                        item.open_time, item.open, item.high, item.low, item.close, item.volume,
                        item.close_time, item.quote_volume, item.count, item.taker_buy_quote_volume,
                        item.taker_buy_base_volume, item.is_final
                    ] for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get klines: {e}")

    def get_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get order book depth for a symbol."""
        try:
            data = self._call(
                'order_book',
                symbol=symbol.upper(),
                limit=limit
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get order book: {e}")

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        try:
            data = self._call(
                'recent_trades',
                symbol=symbol.upper(),
                limit=limit
            )
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get recent trades: {e}")

    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        """Get current mark price for a symbol."""
        try:
            data = self._call(
                'mark_price',
                symbol=symbol.upper()
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get mark price: {e}")

    def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get funding rate history for a symbol."""
        try:
            data = self._call(
                'funding_rates',
                symbol=symbol.upper(),
                limit=limit
            )
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get funding rate: {e}")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders, optionally filtered by symbol."""
        try:
            data = self._call('current_all_open_orders', symbol=symbol.upper() if symbol else None)
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
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
            data = self._call(
                'all_orders',
                symbol=symbol.upper(),
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get all orders: {e}")

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get details of a specific order."""
        try:
            data = self._call(
                'query_order',
                symbol=symbol.upper(),
                order_id=order_id,
                orig_client_order_id=orig_client_order_id
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get order: {e}")

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = TIME_IN_FORCE_GTC,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
        new_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new futures order."""
        try:
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": str(quantity),
            }

            if order_type.upper() == "LIMIT":
                params["time_in_force"] = time_in_force

            if price is not None:
                params["price"] = str(price)

            if stop_price is not None:
                params["stop_price"] = str(stop_price)

            if reduce_only:
                params["reduce_only"] = reduce_only

            if position_side is not None:
                params["position_side"] = position_side.upper()

            if new_client_order_id is not None:
                params["new_client_order_id"] = new_client_order_id

            data = self._call('place_order', **params)
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
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
        time_in_force: str = TIME_IN_FORCE_GTC,
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
            data = self._call(
                'cancel_order',
                symbol=symbol.upper(),
                order_id=order_id,
                orig_client_order_id=orig_client_order_id
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to cancel order: {e}")

    def cancel_all_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        try:
            data = self._call('cancel_all_open_orders', symbol=symbol.upper())
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to cancel all orders: {e}")

    def cancel_orders(
        self,
        symbol: str,
        order_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Cancel multiple orders by their IDs."""
        try:
            data = self._call(
                'cancel_multiple_orders',
                symbol=symbol.upper(),
                order_id_list=order_ids
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to cancel orders: {e}")

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol."""
        try:
            data = self._call(
                'change_initial_leverage',
                symbol=symbol.upper(),
                leverage=leverage
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to set leverage: {e}")

    def set_margin_type(
        self,
        symbol: str,
        margin_type: str,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """Set margin type for a symbol (ISOLATED or CROSSED)."""
        try:
            data = self._call(
                'change_margin_type',
                symbol=symbol.upper(),
                margin_type=margin_type.upper(),
                reduce_only=reduce_only
            )
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to set margin type: {e}")

    def get_leverage_bracket(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get leverage bracket for a symbol or all symbols."""
        try:
            data = self._call('leverage_bracket', symbol=symbol.upper() if symbol else None)
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get leverage bracket: {e}")

    def get_position_mode(self) -> Dict[str, Any]:
        """Get current position mode (hedge mode or one-way)."""
        try:
            data = self._call('get_position_mode')
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get position mode: {e}")

    def set_position_mode(self, hedge_mode: bool) -> Dict[str, Any]:
        """Set position mode (hedge mode or one-way)."""
        try:
            data = self._call('change_position_mode', dual_position_side=hedge_mode)
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
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
            data = self._call(
                'get_income_history',
                symbol=symbol.upper() if symbol else None,
                income_type=income_type,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get income history: {e}")

    def get_notional_and_leverage(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get notional and leverage bracketed info."""
        try:
            data = self._call('notional_and_leverage_bracket', symbol=symbol.upper() if symbol else None)
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
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
            data = self._call(
                'account_trade_list',
                symbol=symbol.upper(),
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            if isinstance(data, list):
                return [self._pydantic_to_dict(item) if hasattr(item, 'model_dump') else item for item in data]
            return data
        except Exception as e:
            raise Exception(f"Failed to get account trades: {e}")

    def get_server_time(self) -> Dict[str, Any]:
        """Get server time."""
        try:
            data = self._call('check_server_time')
            return self._pydantic_to_dict(data) if hasattr(data, 'model_dump') else data
        except Exception as e:
            raise Exception(f"Failed to get server time: {e}")

    def ping(self) -> Dict[str, Any]:
        """Ping the Binance API to check connectivity."""
        try:
            data = self._call('check_server_time')
            return {"status": "ok"}
        except Exception as e:
            raise Exception(f"Failed to ping: {e}")

    @staticmethod
    def _pydantic_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert a Pydantic model to dict."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return obj


def create_binance_client(testnet: bool = True) -> BinanceRESTClient:
    """Convenience function to create a Binance REST client."""
    return BinanceRESTClient(testnet=testnet)
