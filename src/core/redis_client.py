"""
Mimia Quant Trading System - Redis Client
Redis integration for caching and pub/sub messaging
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager

import redis
from redis.exceptions import RedisError


class RedisClient:
    """Redis client wrapper for Mimia Quant trading system"""
    
    # Key prefixes
    PREFIX_MARKET = "market:"
    PREFIX_TRADE = "trade:"
    PREFIX_ORDER = "order:"
    PREFIX_EQUITY = "equity:"
    PREFIX_STRATEGY = "strategy:"
    PREFIX_CONFIG = "config:"
    PREFIX_CACHE = "cache:"
    PREFIX_LOCK = "lock:"
    PREFIX_RATE = "rate:"
    
    # Default TTLs (in seconds)
    TTL_SHORT = 60  # 1 minute
    TTL_MEDIUM = 300  # 5 minutes
    TTL_LONG = 3600  # 1 hour
    TTL_DAY = 86400  # 24 hours
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        decode_responses: bool = True,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        max_connections: int = 10
    ):
        self.host = host or 'localhost'
        self.port = port or 6379
        self.db = db
        self.password = password
        self.decode_responses = decode_responses
        
        self._pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=decode_responses,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            max_connections=max_connections
        )
        self._client = redis.Redis(connection_pool=self._pool)
    
    @property
    def client(self) -> redis.Redis:
        """Get the Redis client instance"""
        return self._client
    
    def ping(self) -> bool:
        """Check if Redis is reachable"""
        try:
            return self._client.ping()
        except RedisError:
            return False
    
    # ==================== Generic Operations ====================
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None, 
        json_encode: bool = True
    ) -> bool:
        """Set a key-value pair with optional TTL"""
        try:
            if json_encode and not isinstance(value, (str, int, float)):
                value = json.dumps(value, default=str)
            return self._client.set(key, value, ex=ttl)
        except RedisError:
            return False
    
    def get(self, key: str, json_decode: bool = True) -> Optional[Any]:
        """Get a value by key"""
        try:
            value = self._client.get(key)
            if value is None:
                return None
            if json_decode:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return value
        except RedisError:
            return None
    
    def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            return self._client.delete(*keys)
        except RedisError:
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if a key exists"""
        try:
            return bool(self._client.exists(key))
        except RedisError:
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time on a key"""
        try:
            return self._client.expire(key, ttl)
        except RedisError:
            return False
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL on a key"""
        try:
            return self._client.ttl(key)
        except RedisError:
            return -1
    
    # ==================== Hash Operations ====================
    
    def hset(self, name: str, key: str, value: Any, json_encode: bool = True) -> int:
        """Set a hash field"""
        try:
            if json_encode and not isinstance(value, (str, int, float)):
                value = json.dumps(value, default=str)
            return self._client.hset(name, key, value)
        except RedisError:
            return 0
    
    def hmset(self, name: str, mapping: Dict[str, Any], json_encode: bool = True) -> bool:
        """Set multiple hash fields"""
        try:
            if json_encode:
                mapping = {
                    k: json.dumps(v, default=str) if not isinstance(v, (str, int, float)) else v
                    for k, v in mapping.items()
                }
            return self._client.hmset(name, mapping)
        except RedisError:
            return False
    
    def hget(self, name: str, key: str, json_decode: bool = True) -> Optional[Any]:
        """Get a hash field"""
        try:
            value = self._client.hget(name, key)
            if value is None:
                return None
            if json_decode:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return value
        except RedisError:
            return None
    
    def hgetall(self, name: str, json_decode: bool = True) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            data = self._client.hgetall(name)
            if not json_decode:
                return data
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            return result
        except RedisError:
            return {}
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        try:
            return self._client.hdel(name, *keys)
        except RedisError:
            return 0
    
    # ==================== List Operations ====================
    
    def lpush(self, name: str, *values: Any, json_encode: bool = True) -> int:
        """Push values to the left of a list"""
        try:
            if json_encode:
                values = [
                    json.dumps(v, default=str) if not isinstance(v, (str, int, float)) else v
                    for v in values
                ]
            return self._client.lpush(name, *values)
        except RedisError:
            return 0
    
    def rpush(self, name: str, *values: Any, json_encode: bool = True) -> int:
        """Push values to the right of a list"""
        try:
            if json_encode:
                values = [
                    json.dumps(v, default=str) if not isinstance(v, (str, int, float)) else v
                    for v in values
                ]
            return self._client.rpush(name, *values)
        except RedisError:
            return 0
    
    def lrange(self, name: str, start: int = 0, end: int = -1, json_decode: bool = True) -> List[Any]:
        """Get list range"""
        try:
            values = self._client.lrange(name, start, end)
            if not json_decode:
                return values
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v)
            return result
        except RedisError:
            return []
    
    def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim a list"""
        try:
            return self._client.ltrim(name, start, end)
        except RedisError:
            return False
    
    # ==================== Sorted Set Operations ====================
    
    def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        """Add members to sorted set"""
        try:
            return self._client.zadd(name, mapping)
        except RedisError:
            return 0
    
    def zrangebyscore(
        self, 
        name: str, 
        min_score: float, 
        max_score: float,
        json_decode: bool = True
    ) -> List[Any]:
        """Get members by score range"""
        try:
            values = self._client.zrangebyscore(name, min_score, max_score)
            if not json_decode:
                return values
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v)
            return result
        except RedisError:
            return []
    
    # ==================== Pub/Sub Operations ====================
    
    def publish(self, channel: str, message: Any, json_encode: bool = True) -> int:
        """Publish a message to a channel"""
        try:
            if json_encode and not isinstance(message, (str, int, float)):
                message = json.dumps(message, default=str)
            return self._client.publish(channel, message)
        except RedisError:
            return 0
    
    @contextmanager
    def pubsub(self):
        """Get a pubsub context manager"""
        pubsub = self._client.pubsub()
        try:
            yield pubsub
        finally:
            pubsub.close()
    
    # ==================== Lock Operations ====================
    
    def acquire_lock(
        self, 
        lock_name: str, 
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: int = None
    ) -> Optional[str]:
        """Acquire a distributed lock"""
        lock_key = f"{self.PREFIX_LOCK}{lock_name}"
        lock_value = f"{datetime.utcnow().timestamp()}"
        
        if blocking:
            end_time = time.time() + (blocking_timeout or timeout)
            while time.time() < end_time:
                if self._client.set(lock_key, lock_value, nx=True, ex=timeout):
                    return lock_value
                time.sleep(0.1)
            return None
        else:
            if self._client.set(lock_key, lock_value, nx=True, ex=timeout):
                return lock_value
            return None
    
    def release_lock(self, lock_name: str, lock_value: str = None) -> bool:
        """Release a distributed lock"""
        lock_key = f"{self.PREFIX_LOCK}{lock_name}"
        if lock_value is None:
            self._client.delete(lock_key)
            return True
        current_value = self._client.get(lock_key)
        if current_value == lock_value:
            self._client.delete(lock_key)
            return True
        return False
    
    # ==================== Rate Limiting ====================
    
    def rate_limit(
        self, 
        key: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Rate limit check using sliding window
        Returns (is_allowed, remaining_requests)
        """
        rate_key = f"{self.PREFIX_RATE}{key}"
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(rate_key, 0, window_start)
        pipe.zcard(rate_key)
        pipe.zadd(rate_key, {str(now): now})
        pipe.expire(rate_key, window_seconds)
        results = pipe.execute()
        
        current_count = results[1]
        if current_count < max_requests:
            return True, max_requests - current_count - 1
        return False, 0
    
    # ==================== Market Data Caching ====================
    
    def cache_market_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        bars: List[Dict],
        ttl: int = TTL_SHORT
    ) -> bool:
        """Cache market bars"""
        key = f"{self.PREFIX_MARKET}{symbol}:{timeframe}"
        return self.set(key, bars, ttl=ttl)
    
    def get_cached_market_bars(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Optional[List[Dict]]:
        """Get cached market bars"""
        key = f"{self.PREFIX_MARKET}{symbol}:{timeframe}"
        return self.get(key)
    
    # ==================== Order Book Caching ====================
    
    def cache_order_book(
        self, 
        symbol: str, 
        order_book: Dict,
        ttl: int = TTL_SHORT
    ) -> bool:
        """Cache order book"""
        key = f"{self.PREFIX_MARKET}orderbook:{symbol}"
        return self.hmset(key, order_book, json_encode=True) and self.expire(key, ttl)
    
    def get_cached_order_book(self, symbol: str) -> Optional[Dict]:
        """Get cached order book"""
        key = f"{self.PREFIX_MARKET}orderbook:{symbol}"
        return self.hgetall(key, json_decode=True)
    
    # ==================== Strategy State ====================
    
    def save_strategy_state(
        self, 
        strategy_name: str, 
        session_id: str,
        state: Dict,
        ttl: int = TTL_DAY
    ) -> bool:
        """Save strategy state"""
        key = f"{self.PREFIX_STRATEGY}{strategy_name}:{session_id}"
        return self.hmset(key, state, json_encode=True) and self.expire(key, ttl)
    
    def get_strategy_state(
        self, 
        strategy_name: str, 
        session_id: str
    ) -> Optional[Dict]:
        """Get strategy state"""
        key = f"{self.PREFIX_STRATEGY}{strategy_name}:{session_id}"
        return self.hgetall(key, json_decode=True)
    
    # ==================== Config Caching ====================
    
    def cache_config(
        self, 
        config_name: str, 
        config: Dict,
        ttl: int = TTL_LONG
    ) -> bool:
        """Cache configuration"""
        key = f"{self.PREFIX_CONFIG}{config_name}"
        return self.set(key, config, ttl=ttl, json_encode=True)
    
    def get_cached_config(self, config_name: str) -> Optional[Dict]:
        """Get cached configuration"""
        key = f"{self.PREFIX_CONFIG}{config_name}"
        return self.get(key)
    
    # ==================== Generic Cache ====================
    
    def cache_result(
        self, 
        cache_key: str, 
        result: Any,
        ttl: int = TTL_MEDIUM
    ) -> bool:
        """Generic result caching"""
        key = f"{self.PREFIX_CACHE}{cache_key}"
        return self.set(key, result, ttl=ttl)
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        key = f"{self.PREFIX_CACHE}{cache_key}"
        return self.get(key)
    
    def invalidate_cache(self, cache_key: str) -> int:
        """Invalidate cache entries by pattern"""
        pattern = f"{self.PREFIX_CACHE}{cache_key}"
        keys = self._client.keys(pattern)
        if keys:
            return self.delete(*keys)
        return 0
    
    # ==================== Cleanup ====================
    
    def flush_pattern(self, pattern: str) -> int:
        """Flush all keys matching a pattern"""
        keys = self._client.keys(pattern)
        if keys:
            return self.delete(*keys)
        return 0
    
    def close(self):
        """Close Redis connection"""
        self._pool.disconnect()


class RedisManager:
    """Singleton Redis manager"""
    _instance: Optional[RedisClient] = None
    
    @classmethod
    def get_instance(
        cls, 
        host: str = None, 
        port: int = None, 
        **kwargs
    ) -> RedisClient:
        """Get or create Redis client instance"""
        if cls._instance is None:
            cls._instance = RedisClient(host=host, port=port, **kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance"""
        if cls._instance:
            cls._instance.close()
            cls._instance = None


import time


def get_redis_client(
    host: str = None,
    port: int = None,
    **kwargs
) -> RedisClient:
    """Factory function to get Redis client"""
    return RedisManager.get_instance(host=host, port=port, **kwargs)
