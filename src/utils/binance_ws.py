"""
Binance WebSocket Client for Mimia Quant Trading System.
Supports testnet and real Binance Futures WebSocket streams.
"""

import json
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from websocket import (
    WebSocketApp,
    WebSocketConnectionClosedException,
    enableTrace
)


class BinanceWebSocketClient:
    """
    Binance Futures WebSocket client with testnet support.
    Supports multiple streams: klines, ticker, trades, book depth, and user data.
    """
    
    STREAM_URL = "wss://stream.binancefuture.com/ws"
    TESTNET_STREAM_URL = "wss://stream.binancefuture.com/ws"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        enable_trace: bool = False
    ):
        """
        Initialize Binance WebSocket client.
        
        Args:
            api_key: Binance API key (for user data stream)
            api_secret: Binance API secret
            testnet: Use testnet (default True)
            on_message: Callback for received messages
            on_error: Callback for errors
            on_close: Callback when connection closes
            on_open: Callback when connection opens
            enable_trace: Enable WebSocket tracing
        """
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        
        self.stream_url = self.TESTNET_STREAM_URL if testnet else self.STREAM_URL
        
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscriptions: List[str] = []
        self._lock = threading.Lock()
        
        self._on_message_callback = on_message
        self._on_error_callback = on_error
        self._on_close_callback = on_close
        self._on_open_callback = on_open
        
        if enable_trace:
            enableTrace(True)
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._running
    
    def _get_message_handler(self) -> Callable:
        """Create the message handler wrapper."""
        def handle_message(ws: WebSocketApp, message: str):
            try:
                data = json.loads(message)
                
                # Handle subscription responses
                if "result" in data and data.get("id") is not None:
                    pass  # Subscription confirmation
                
                # Handle stream data
                elif "e" in data:  # Event type present
                    event_type = data.get("e")
                    
                    # Route to specific handler if registered
                    handler = self._stream_handlers.get(event_type)
                    if handler:
                        handler(data)
                
                # Call user callback if provided
                if self._on_message_callback:
                    self._on_message_callback(data)
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse message: {e}")
            except Exception as e:
                print(f"Error handling message: {e}")
        
        return handle_message
    
    def _get_error_handler(self) -> Callable:
        """Create the error handler wrapper."""
        def handle_error(ws: WebSocketApp, error):
            if self._on_error_callback:
                self._on_error_callback(error)
            else:
                print(f"WebSocket error: {error}")
        
        return handle_error
    
    def _get_close_handler(self) -> Callable:
        """Create the close handler wrapper."""
        def handle_close(ws: WebSocketApp, close_status_code: int, close_msg: str):
            self._running = False
            if self._on_close_callback:
                self._on_close_callback(close_status_code, close_msg)
            else:
                print(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        return handle_close
    
    def _get_open_handler(self) -> Callable:
        """Create the open handler wrapper."""
        def handle_open(ws: WebSocketApp):
            # Resubscribe to previous streams
            if self._subscriptions:
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": list(set(self._subscriptions)),
                    "id": int(time.time() * 1000)
                }
                ws.send(json.dumps(subscribe_msg))
            
            if self._on_open_callback:
                self._on_open_callback()
            else:
                print("WebSocket connected")
        
        return handle_open
    
    @property
    def _stream_handlers(self) -> Dict[str, Callable]:
        """Dictionary of stream event handlers."""
        return getattr(self, '_stream_handlers_dict', {})
    
    def connect(self, streams: Optional[List[str]] = None):
        """
        Connect to Binance WebSocket.
        
        Args:
            streams: List of stream names to subscribe to
        """
        if self._running:
            print("WebSocket already connected")
            return
        
        self._running = True
        self._ws = WebSocketApp(
            self.stream_url,
            on_message=self._get_message_handler(),
            on_error=self._get_error_handler(),
            on_close=self._get_close_handler(),
            on_open=self._get_open_handler()
        )
        
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        
        # Subscribe to streams
        if streams:
            self.subscribe(streams)
    
    def disconnect(self):
        """Disconnect from Binance WebSocket."""
        self._running = False
        if self._ws:
            self._ws.close()
            self._ws = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
    
    def _send(self, data: Dict[str, Any]):
        """Send data through WebSocket."""
        if self._ws and self._running:
            try:
                self._ws.send(json.dumps(data))
            except WebSocketConnectionClosedException:
                print("WebSocket connection closed")
        else:
            print("WebSocket not connected")
    
    def subscribe(self, streams: List[str], subscription_id: Optional[int] = None):
        """
        Subscribe to WebSocket streams.
        
        Args:
            streams: List of stream names (e.g., ['btcusdt@kline_1m', 'btcusdt@ticker'])
            subscription_id: Optional custom subscription ID
        """
        if not streams:
            return
        
        with self._lock:
            # Add to subscriptions list (remove any existing first)
            for stream in streams:
                if stream not in self._subscriptions:
                    self._subscriptions.append(stream)
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": list(set(streams)),
            "id": subscription_id or int(time.time() * 1000)
        }
        self._send(subscribe_msg)
    
    def unsubscribe(self, streams: List[str], unsubscription_id: Optional[int] = None):
        """
        Unsubscribe from WebSocket streams.
        
        Args:
            streams: List of stream names to unsubscribe from
            unsubscription_id: Optional custom unsubscription ID
        """
        with self._lock:
            for stream in streams:
                if stream in self._subscriptions:
                    self._subscriptions.remove(stream)
        
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": list(set(streams)),
            "id": unsubscription_id or int(time.time() * 1000)
        }
        self._send(unsubscribe_msg)
    
    def subscribe_kline(
        self,
        symbol: str,
        interval: str = "1m",
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to kline/candlestick stream for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        self.subscribe([stream], subscription_id)
    
    def subscribe_ticker(
        self,
        symbol: str,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to 24hr rolling window ticker stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@ticker"
        self.subscribe([stream], subscription_id)
    
    def subscribe_all_tickers(self, subscription_id: Optional[int] = None):
        """
        Subscribe to all market 24hr rolling window tickers.
        
        Args:
            subscription_id: Optional custom subscription ID
        """
        stream = "!ticker@arr"
        self.subscribe([stream], subscription_id)
    
    def subscribe_trade(
        self,
        symbol: str,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to trade stream for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@trade"
        self.subscribe([stream], subscription_id)
    
    def subscribe_book_depth(
        self,
        symbol: str,
        limit: int = 100,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to partial book depth stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            limit: Depth levels (5, 10, 20, 50, 100, 500, 1000)
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@depth{limit}"
        self.subscribe([stream], subscription_id)
    
    def subscribe_book_ticker(
        self,
        symbol: str,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to book ticker (best bid/ask) stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@bookTicker"
        self.subscribe([stream], subscription_id)
    
    def subscribe_mark_price(
        self,
        symbol: str,
        update_interval: int = 1,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to mark price stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            update_interval: Update interval in seconds (1 or 3)
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@markPrice@{update_interval}s"
        self.subscribe([stream], subscription_id)
    
    def subscribe_funding_rate(
        self,
        symbol: str,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to funding rate stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@fundingRate"
        self.subscribe([stream], subscription_id)
    
    def subscribe_composite_index(
        self,
        symbol: str,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to composite index stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@compositeIndex"
        self.subscribe([stream], subscription_id)
    
    def subscribe_liquidation_stream(
        self,
        symbol: Optional[str] = None,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to liquidation stream.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt'), or None for all symbols
            subscription_id: Optional custom subscription ID
        """
        if symbol:
            stream = f"{symbol.lower()}@forceOrder"
        else:
            stream = "!forceOrder@arr"
        self.subscribe([stream], subscription_id)
    
    def subscribe_diff_book_depth(
        self,
        symbol: str,
        limit: int = 100,
        subscription_id: Optional[int] = None
    ):
        """
        Subscribe to diff book depth stream (updates only).
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            limit: Depth levels (5, 10, 20, 50, 100, 500, 1000)
            subscription_id: Optional custom subscription ID
        """
        stream = f"{symbol.lower()}@depth@100ms"
        self.subscribe([stream], subscription_id)
    
    def register_kline_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for kline events."""
        self._stream_handlers_dict["kline"] = handler
    
    def register_ticker_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for ticker events."""
        self._stream_handlers_dict["24hrTicker"] = handler
    
    def register_trade_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for trade events."""
        self._stream_handlers_dict["trade"] = handler
    
    def register_book_depth_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for book depth events."""
        self._stream_handlers_dict["depthUpdate"] = handler
    
    def register_book_ticker_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for book ticker events."""
        self._stream_handlers_dict["bookTicker"] = handler
    
    def register_mark_price_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for mark price events."""
        self._stream_handlers_dict["markPriceUpdate"] = handler
    
    def register_funding_rate_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for funding rate events."""
        self._stream_handlers_dict["fundingRate"] = handler
    
    def register_order_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for order update events (ACCOUNT_UPDATE)."""
        self._stream_handlers_dict["ACCOUNT_UPDATE"] = handler
    
    def register_orderTrade_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for order trade events (ORDER_TRADE_UPDATE)."""
        self._stream_handlers_dict["ORDER_TRADE_UPDATE"] = handler
    
    def register_account_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for account update events."""
        self._stream_handlers_dict["ACCOUNT_UPDATE"] = handler
    
    def register_trade_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for trade update events."""
        self._stream_handlers_dict["ORDER_TRADE_UPDATE"] = handler
    
    def register_margin_call_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for margin call events."""
        self._stream_handlers_dict["MARGIN_CALL"] = handler
    
    def register_balance_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for balance update events."""
        self._stream_handlers_dict["BALANCE"] = handler
    
    def register_position_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for position update events."""
        self._stream_handlers_dict["POSITION"] = handler
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


class BinanceUserDataWebSocket:
    """
    Binance Futures User Data WebSocket stream.
    Provides real-time updates for account balance, positions, and orders.
    """
    
    USER_DATA_STREAM_URL = "wss://stream.binancefuture.com/ws"
    TESTNET_USER_DATA_STREAM_URL = "wss://stream.binancefuture.com/ws"
    
    def __init__(
        self,
        listen_key: str,
        testnet: bool = True,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_open: Optional[Callable] = None
    ):
        """
        Initialize User Data WebSocket client.
        
        Args:
            listen_key: User data stream listen key (from REST API)
            testnet: Use testnet (default True)
            on_message: Callback for received messages
            on_error: Callback for errors
            on_close: Callback when connection closes
            on_open: Callback when connection opens
        """
        self.listen_key = listen_key
        self.testnet = testnet
        self.stream_url = self.TESTNET_USER_DATA_STREAM_URL if testnet else self.USER_DATA_STREAM_URL
        
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        self._on_message_callback = on_message
        self._on_error_callback = on_error
        self._on_close_callback = on_close
        self._on_open_callback = on_open
        
        self._stream_handlers_dict: Dict[str, Callable] = {}
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._running
    
    def _get_message_handler(self) -> Callable:
        """Create the message handler wrapper."""
        def handle_message(ws: WebSocketApp, message: str):
            try:
                data = json.loads(message)
                
                # Handle subscription responses
                if "result" in data and data.get("id") is not None:
                    pass  # Subscription confirmation
                
                # Handle stream data
                elif "e" in data:  # Event type present
                    event_type = data.get("e")
                    handler = self._stream_handlers_dict.get(event_type)
                    if handler:
                        handler(data)
                
                if self._on_message_callback:
                    self._on_message_callback(data)
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse message: {e}")
            except Exception as e:
                print(f"Error handling message: {e}")
        
        return handle_message
    
    def _get_error_handler(self) -> Callable:
        """Create the error handler wrapper."""
        def handle_error(ws: WebSocketApp, error):
            if self._on_error_callback:
                self._on_error_callback(error)
            else:
                print(f"User WebSocket error: {error}")
        
        return handle_error
    
    def _get_close_handler(self) -> Callable:
        """Create the close handler wrapper."""
        def handle_close(ws: WebSocketApp, close_status_code: int, close_msg: str):
            self._running = False
            if self._on_close_callback:
                self._on_close_callback(close_status_code, close_msg)
            else:
                print(f"User WebSocket closed: {close_status_code} - {close_msg}")
        
        return handle_close
    
    def _get_open_handler(self) -> Callable:
        """Create the open handler wrapper."""
        def handle_open(ws: WebSocketApp):
            # Subscribe to user data stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [self.listen_key],
                "id": int(time.time() * 1000)
            }
            ws.send(json.dumps(subscribe_msg))
            
            if self._on_open_callback:
                self._on_open_callback()
            else:
                print("User WebSocket connected")
        
        return handle_open
    
    def connect(self):
        """Connect to User Data WebSocket stream."""
        if self._running:
            print("User WebSocket already connected")
            return
        
        self._running = True
        self._ws = WebSocketApp(
            self.stream_url,
            on_message=self._get_message_handler(),
            on_error=self._get_error_handler(),
            on_close=self._get_close_handler(),
            on_open=self._get_open_handler()
        )
        
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
    
    def disconnect(self):
        """Disconnect from User Data WebSocket."""
        self._running = False
        if self._ws:
            self._ws.close()
            self._ws = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
    
    def register_account_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for ACCOUNT_UPDATE events."""
        self._stream_handlers_dict["ACCOUNT_UPDATE"] = handler
    
    def register_order_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for ORDER_TRADE_UPDATE events."""
        self._stream_handlers_dict["ORDER_TRADE_UPDATE"] = handler
    
    def register_margin_call_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for MARGIN_CALL events."""
        self._stream_handlers_dict["MARGIN_CALL"] = handler
    
    def register_listen_key_expired_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for LISTEN_KEY_EXPIRED events."""
        self._stream_handlers_dict["LISTEN_KEY_EXPIRED"] = handler
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# Convenience function to get listen key
def get_listen_key(client) -> str:
    """
    Get a listen key for user data stream.
    
    Args:
        client: BinanceRESTClient instance
    
    Returns:
        Listen key string
    """
    return client.client.futures_account().get("listenKey", "")


# Convenience function to create user data WebSocket
def create_user_data_websocket(
    listen_key: str,
    testnet: bool = True,
    on_message: Optional[Callable] = None,
    on_error: Optional[Callable] = None
) -> BinanceUserDataWebSocket:
    """Create and return a User Data WebSocket client."""
    return BinanceUserDataWebSocket(
        listen_key=listen_key,
        testnet=testnet,
        on_message=on_message,
        on_error=on_error
    )
