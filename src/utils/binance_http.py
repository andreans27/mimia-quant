"""
Binance HTTP Client using requests directly.
This bypasses python-binance's issues with demo-fapi.binance.com.

Per Binance docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
"""

import hmac
import hashlib
import time
import requests
from typing import Optional, Dict, Any, List


class BinanceHTTPClient:
    """
    Binance Futures HTTP client using requests directly.
    This resolves python-binance compatibility issues with demo-fapi.binance.com.
    
    NOTE: demo-fapi.binance.com only supports public endpoints.
    For authenticated trading, use testnet.binancefuture.com or production.
    """
    
    # REST API Base URLs
    # - demo-fapi.binance.com: Public market data only (no trading)
    # - testnet.binancefuture.com: Trading API (requires API key from testnet portal)
    # - fapi.binance.com: Production (real money)
    DEMO_FUTURES_URL = "https://demo-fapi.binance.com"
    TESTNET_FUTURES_URL = "https://testnet.binancefuture.com"
    REAL_FUTURES_URL = "https://fapi.binance.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        trading: bool = False  # If True, use testnet.binancefuture.com for trading
    ):
        """
        Initialize Binance HTTP client.
        
        Args:
            api_key: Binance API key (or env BINANCE_API_KEY)
            api_secret: Binance API secret (or env BINANCE_API_SECRET)
            testnet: Use testnet/demo (default True)
            trading: If True, use testnet.binancefuture.com for trading API.
                    If False, use demo-fapi.binance.com for market data only.
        """
        import os
        
        self.testnet = testnet
        
        # Determine base URL based on testnet and trading flags
        if not testnet:
            # Production
            self.base_url = self.REAL_FUTURES_URL
        elif trading:
            # Testnet trading API (testnet.binancefuture.com)
            self.base_url = self.TESTNET_FUTURES_URL
        else:
            # Demo market data only (demo-fapi.binance.com)
            self.base_url = self.DEMO_FUTURES_URL
        
        if testnet:
            self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        recvWindow: int = 5000
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Binance API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path (e.g., '/fapi/v1/account')
            signed: Whether request requires signature
            params: Request parameters
            recvWindow: recvWindow in milliseconds (default 5000)
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = recvWindow
            params["signature"] = self._generate_signature(params)
        
        if method == "GET":
            response = self.session.get(url, params=params, timeout=10)
        elif method == "POST":
            response = self.session.post(url, data=params, timeout=10)
        elif method == "DELETE":
            response = self.session.delete(url, params=params, timeout=10)
        elif method == "PUT":
            response = self.session.put(url, data=params, timeout=10)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code != 200:
            raise Exception(f"Binance API error {response.status_code}: {response.text}")
        
        return response.json()
    
    # ==================== PUBLIC ENDPOINTS ====================
    
    def ping(self) -> Dict[str, Any]:
        """Ping the API."""
        return self._request("GET", "/fapi/v1/ping")
    
    def get_server_time(self) -> Dict[str, Any]:
        """Get server time."""
        return self._request("GET", "/fapi/v1/time")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        return self._request("GET", "/fapi/v1/exchangeInfo")
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List[Any]]:
        """Get candlestick/kline data."""
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._request("GET", "/fapi/v1/klines", params=params)
    
    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        """Get current mark price for a symbol."""
        return self._request("GET", "/fapi/v1/markPrice", params={"symbol": symbol.upper()})
    
    def get_funding_rate(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get funding rate history."""
        params = {"symbol": symbol.upper(), "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._request("GET", "/fapi/v1/fundingRate", params=params)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book depth."""
        return self._request("GET", "/fapi/v1/depth", params={"symbol": symbol.upper(), "limit": limit})
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent trades."""
        return self._request("GET", "/fapi/v1/trades", params={"symbol": symbol.upper(), "limit": limit})
    
    # ==================== AUTHENTICATED ENDPOINTS ====================
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return self._request("GET", "/fapi/v1/account", signed=True)
    
    def get_balance(self) -> List[Dict[str, Any]]:
        """Get futures account balance."""
        return self._request("GET", "/fapi/v1/balance", signed=True)
    
    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get position information."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/fapi/v1/positionRisk", signed=True, params=params)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/fapi/v1/openOrders", signed=True, params=params)
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new order."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "timeInForce": time_in_force,
            "reduceOnly": str(reduce_only).lower()
        }
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if position_side is not None:
            params["positionSide"] = position_side.upper()
        
        return self._request("POST", "/fapi/v1/order", signed=True, params=params)
    
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an order."""
        params = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        
        return self._request("DELETE", "/fapi/v1/order", signed=True, params=params)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol."""
        return self._request(
            "POST", "/fapi/v1/leverage",
            signed=True,
            params={"symbol": symbol.upper(), "leverage": leverage}
        )
    
    def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get income history."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if income_type:
            params["incomeType"] = income_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        return self._request("GET", "/fapi/v1/income", signed=True, params=params)


# Convenience function for quick client creation
def create_binance_http_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = True
) -> BinanceHTTPClient:
    """Create a BinanceHTTPClient instance."""
    return BinanceHTTPClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )


if __name__ == "__main__":
    # Test the client
    import os
    os.environ["BINANCE_TESTNET_API_KEY"] = "D3Wbkd8THmr8ZLGfB4NGtTVBlJcorYFeOCrCOltebRcPDIeHVekDssJFogzzaGX4"
    os.environ["BINANCE_TESTNET_API_SECRET"] = "lyAK3TKEatiRnQvwLV0ixBkfYPm1surlxythQ7j7pjZdWP2fcK9JHPtTPMOmTcoAx"
    
    client = BinanceHTTPClient(testnet=True)
    print("Base URL:", client.base_url)
    print()
    
    # Public tests
    print("--- Public Endpoints ---")
    print("Ping:", client.ping())
    
    exchange_info = client.get_exchange_info()
    print("Exchange symbols:", len(exchange_info.get("symbols", [])))
    
    klines = client.get_klines("BTCUSDT", "1h", limit=5)
    print("Klines (BTCUSDT 1h):", len(klines), "bars")
    if klines:
        print("  Latest close:", klines[-1][4])
    
    # Auth tests
    print()
    print("--- Authenticated Endpoints ---")
    try:
        balance = client.get_balance()
        print("Balance:", balance)
    except Exception as e:
        print("Balance error:", e)
    
    try:
        account = client.get_account_info()
        print("Account totalBalance:", account.get("totalBalance", "N/A"))
    except Exception as e:
        print("Account error:", e)
