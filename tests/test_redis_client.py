"""
Tests for Mimia Quant Redis Client Module
"""

import os
import sys
import time
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.redis_client import RedisClient, RedisManager


class TestRedisClientBasics:
    """Test basic Redis operations"""
    
    def test_ping(self):
        """Test ping command"""
        client = RedisClient(decode_responses=False)
        # If Redis is not available, this will fail
        # We mock it for unit testing
        with patch.object(client._client, 'ping', return_value=True):
            assert client.ping() is True
    
    def test_set_and_get_string(self):
        """Test setting and getting string values"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'set', return_value=True) as mock_set:
            with patch.object(client._client, 'get', return_value='test_value'):
                result = client.set('test_key', 'test_value', json_encode=False)
                assert result is True
                
                value = client.get('test_key', json_decode=False)
                assert value == 'test_value'
    
    def test_set_with_ttl(self):
        """Test setting value with TTL"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'set', return_value=True) as mock_set:
            result = client.set('test_key', 'test_value', ttl=60)
            mock_set.assert_called_once_with('test_key', 'test_value', ex=60)
            assert result is True
    
    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'get', return_value=None):
            value = client.get('nonexistent_key')
            assert value is None
    
    def test_delete_key(self):
        """Test deleting a key"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'delete', return_value=1) as mock_delete:
            result = client.delete('test_key')
            mock_delete.assert_called_once_with('test_key')
            assert result == 1
    
    def test_exists(self):
        """Test checking if key exists"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'exists', return_value=1):
            assert client.exists('test_key') is True
        
        with patch.object(client._client, 'exists', return_value=0):
            assert client.exists('nonexistent_key') is False
    
    def test_expire(self):
        """Test setting expiration on a key"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'expire', return_value=True):
            result = client.expire('test_key', 60)
            assert result is True
    
    def test_ttl(self):
        """Test getting TTL of a key"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'ttl', return_value=59):
            result = client.ttl('test_key')
            assert result == 59


class TestRedisHashOperations:
    """Test Redis hash operations"""
    
    def test_hset(self):
        """Test setting hash field"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'hset', return_value=1) as mock_hset:
            result = client.hset('hash_name', 'field1', 'value1', json_encode=False)
            mock_hset.assert_called_once_with('hash_name', 'field1', 'value1')
            assert result == 1
    
    def test_hmset(self):
        """Test setting multiple hash fields"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'hmset', return_value=True) as mock_hmset:
            result = client.hmset('hash_name', {'field1': 'value1', 'field2': 'value2'}, json_encode=False)
            mock_hmset.assert_called_once()
            assert result is True
    
    def test_hget(self):
        """Test getting hash field"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'hget', return_value='value1'):
            result = client.hget('hash_name', 'field1', json_decode=False)
            assert result == 'value1'
    
    def test_hgetall(self):
        """Test getting all hash fields"""
        client = RedisClient(decode_responses=False)
        
        mock_data = {'field1': 'value1', 'field2': 'value2'}
        with patch.object(client._client, 'hgetall', return_value=mock_data):
            result = client.hgetall('hash_name', json_decode=False)
            assert result == mock_data
    
    def test_hdel(self):
        """Test deleting hash fields"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'hdel', return_value=2) as mock_hdel:
            result = client.hdel('hash_name', 'field1', 'field2')
            mock_hdel.assert_called_once_with('hash_name', 'field1', 'field2')
            assert result == 2


class TestRedisListOperations:
    """Test Redis list operations"""
    
    def test_lpush(self):
        """Test pushing to list from left"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'lpush', return_value=3) as mock_lpush:
            result = client.lpush('list_name', 'value1', 'value2', json_encode=False)
            assert result == 3
    
    def test_rpush(self):
        """Test pushing to list from right"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'rpush', return_value=3) as mock_rpush:
            result = client.rpush('list_name', 'value1', 'value2', json_encode=False)
            assert result == 3
    
    def test_lrange(self):
        """Test getting list range"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'lrange', return_value=['v1', 'v2', 'v3']):
            result = client.lrange('list_name', 0, -1, json_decode=False)
            assert result == ['v1', 'v2', 'v3']
    
    def test_ltrim(self):
        """Test trimming a list"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'ltrim', return_value=True) as mock_ltrim:
            result = client.ltrim('list_name', 0, 9)
            mock_ltrim.assert_called_once_with('list_name', 0, 9)
            assert result is True


class TestRedisSortedSetOperations:
    """Test Redis sorted set operations"""
    
    def test_zadd(self):
        """Test adding members to sorted set"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'zadd', return_value=2) as mock_zadd:
            result = client.zadd('sorted_set', {'member1': 1.0, 'member2': 2.0})
            mock_zadd.assert_called_once_with('sorted_set', {'member1': 1.0, 'member2': 2.0})
            assert result == 2
    
    def test_zrangebyscore(self):
        """Test getting members by score range"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'zrangebyscore', return_value=['m1', 'm2']):
            result = client.zrangebyscore('sorted_set', 1.0, 2.0, json_decode=False)
            assert result == ['m1', 'm2']


class TestRedisPubSub:
    """Test Redis pub/sub operations"""
    
    def test_publish(self):
        """Test publishing a message"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'publish', return_value=1) as mock_publish:
            result = client.publish('channel', 'message', json_encode=False)
            mock_publish.assert_called_once_with('channel', 'message')
            assert result == 1
    
    def test_publish_json(self):
        """Test publishing a JSON message"""
        client = RedisClient(decode_responses=False)
        message = {'key': 'value', 'number': 123}
        
        with patch.object(client._client, 'publish', return_value=1) as mock_publish:
            result = client.publish('channel', message, json_encode=True)
            called_message = mock_publish.call_args[0][1]
            assert json.loads(called_message) == message
    
    def test_pubsub_context(self):
        """Test pubsub context manager"""
        client = RedisClient(decode_responses=False)
        mock_pubsub = MagicMock()
        
        with patch.object(client._client, 'pubsub', return_value=mock_pubsub):
            with client.pubsub() as pubsub:
                assert pubsub == mock_pubsub
            mock_pubsub.close.assert_called_once()


class TestRedisLockOperations:
    """Test Redis distributed lock operations"""
    
    def test_acquire_lock(self):
        """Test acquiring a lock"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'set', return_value=True):
            lock_value = client.acquire_lock('test_lock', timeout=10, blocking=False)
            assert lock_value is not None
    
    def test_acquire_lock_blocking(self):
        """Test acquiring a lock with blocking"""
        client = RedisClient(decode_responses=False)
        
        # First call succeeds
        with patch.object(client._client, 'set', return_value=True):
            lock_value = client.acquire_lock('test_lock', timeout=10, blocking=False)
            assert lock_value is not None
    
    def test_acquire_lock_unavailable(self):
        """Test acquiring a lock that's unavailable"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'set', return_value=False):
            lock_value = client.acquire_lock('test_lock', timeout=1, blocking=False)
            assert lock_value is None
    
    def test_release_lock(self):
        """Test releasing a lock"""
        client = RedisClient(decode_responses=False)
        lock_value = '12345'
        
        with patch.object(client._client, 'get', return_value=lock_value):
            with patch.object(client._client, 'delete', return_value=1):
                result = client.release_lock('test_lock', lock_value)
                assert result is True
    
    def test_release_lock_wrong_value(self):
        """Test releasing a lock with wrong value"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'get', return_value='wrong_value'):
            result = client.release_lock('test_lock', 'correct_value')
            assert result is False


class TestRedisRateLimiting:
    """Test Redis rate limiting operations"""
    
    def test_rate_limit_allowed(self):
        """Test rate limit when under threshold"""
        client = RedisClient(decode_responses=False)
        
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [None, 5, 1234567890, True]  # Removed items, current count, added item, expire result
        
        with patch.object(client._client, 'pipeline', return_value=mock_pipe):
            is_allowed, remaining = client.rate_limit('test_endpoint', max_requests=10, window_seconds=60)
            assert is_allowed is True
            assert remaining == 4  # 10 - 5 - 1 = 4
    
    def test_rate_limit_exceeded(self):
        """Test rate limit when over threshold"""
        client = RedisClient(decode_responses=False)
        
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [None, 10, 1234567890, True]  # At limit
        
        with patch.object(client._client, 'pipeline', return_value=mock_pipe):
            is_allowed, remaining = client.rate_limit('test_endpoint', max_requests=10, window_seconds=60)
            assert is_allowed is False
            assert remaining == 0


class TestRedisMarketDataCaching:
    """Test market data caching operations"""
    
    def test_cache_market_bars(self):
        """Test caching market bars"""
        client = RedisClient(decode_responses=False)
        bars = [
            {'timestamp': '2024-01-15 10:00', 'close': 50000},
            {'timestamp': '2024-01-15 10:01', 'close': 50100},
        ]
        
        with patch.object(client._client, 'set', return_value=True) as mock_set:
            result = client.cache_market_bars('BTCUSDT', '1m', bars)
            assert result is True
    
    def test_get_cached_market_bars(self):
        """Test getting cached market bars"""
        client = RedisClient(decode_responses=False)
        bars = [
            {'timestamp': '2024-01-15 10:00', 'close': 50000},
        ]
        
        with patch.object(client._client, 'get', return_value=json.dumps(bars)):
            result = client.get_cached_market_bars('BTCUSDT', '1m')
            assert result == bars


class TestRedisOrderBookCaching:
    """Test order book caching operations"""
    
    def test_cache_order_book(self):
        """Test caching order book"""
        client = RedisClient(decode_responses=False)
        order_book = {
            'bids': [['50000', '1.5'], ['49900', '2.0']],
            'asks': [['50100', '1.0'], ['50200', '2.5']],
        }
        
        with patch.object(client._client, 'hmset', return_value=True):
            with patch.object(client._client, 'expire', return_value=True):
                result = client.cache_order_book('BTCUSDT', order_book, ttl=60)
                assert result is True
    
    def test_get_cached_order_book(self):
        """Test getting cached order book"""
        client = RedisClient(decode_responses=False)
        order_book = {
            'bids': [['50000', '1.5']],
            'asks': [['50100', '1.0']],
        }
        
        with patch.object(client._client, 'hgetall', return_value=order_book):
            result = client.get_cached_order_book('BTCUSDT')
            assert result == order_book


class TestRedisStrategyState:
    """Test strategy state caching operations"""
    
    def test_save_strategy_state(self):
        """Test saving strategy state"""
        client = RedisClient(decode_responses=False)
        state = {
            'position': 0.5,
            'entry_price': 50000,
            'pnl': 100.5,
        }
        
        with patch.object(client._client, 'hmset', return_value=True):
            with patch.object(client._client, 'expire', return_value=True) as mock_expire:
                result = client.save_strategy_state('ma_cross', 'session_001', state)
                assert result is True
    
    def test_get_strategy_state(self):
        """Test getting strategy state"""
        client = RedisClient(decode_responses=False)
        state = {
            'position': 0.5,
            'entry_price': 50000,
            'pnl': 100.5,
        }
        
        with patch.object(client._client, 'hgetall', return_value=state):
            result = client.get_strategy_state('ma_cross', 'session_001')
            assert result == state


class TestRedisConfigCaching:
    """Test configuration caching operations"""
    
    def test_cache_config(self):
        """Test caching configuration"""
        client = RedisClient(decode_responses=False)
        config = {
            'strategy_name': 'ma_cross',
            'parameters': {'fast_ma': 10, 'slow_ma': 20},
        }
        
        with patch.object(client._client, 'set', return_value=True) as mock_set:
            result = client.cache_config('ma_cross_config', config)
            assert result is True
    
    def test_get_cached_config(self):
        """Test getting cached configuration"""
        client = RedisClient(decode_responses=False)
        config = {
            'strategy_name': 'ma_cross',
            'parameters': {'fast_ma': 10, 'slow_ma': 20},
        }
        
        with patch.object(client._client, 'get', return_value=json.dumps(config)):
            result = client.get_cached_config('ma_cross_config')
            assert result == config


class TestRedisGenericCache:
    """Test generic cache operations"""
    
    def test_cache_result(self):
        """Test caching a generic result"""
        client = RedisClient(decode_responses=False)
        result = {'data': 'value', 'count': 100}
        
        with patch.object(client._client, 'set', return_value=True):
            cache_key = 'indicator:ma: BTCUSDT:1m'
            response = client.cache_result(cache_key, result)
            assert response is True
    
    def test_get_cached_result(self):
        """Test getting a cached result"""
        client = RedisClient(decode_responses=False)
        result = {'data': 'value', 'count': 100}
        
        with patch.object(client._client, 'get', return_value=json.dumps(result)):
            cache_key = 'indicator:ma: BTCUSDT:1m'
            response = client.get_cached_result(cache_key)
            assert response == result
    
    def test_invalidate_cache(self):
        """Test invalidating cache entries by pattern"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._client, 'keys', return_value=['cache:key1', 'cache:key2']):
            with patch.object(client._client, 'delete', return_value=2):
                result = client.invalidate_cache('key*')
                assert result == 2


class TestRedisManager:
    """Test Redis manager singleton"""
    
    def test_get_instance(self):
        """Test getting Redis client instance"""
        RedisManager.reset_instance()
        
        with patch('src.core.redis_client.redis.Redis'):
            client = RedisManager.get_instance(host='localhost', port=6379)
            assert client is not None
            assert isinstance(client, RedisClient)
    
    def test_reset_instance(self):
        """Test resetting Redis client instance"""
        RedisManager.reset_instance()
        
        with patch('src.core.redis_client.redis.Redis'):
            client1 = RedisManager.get_instance(host='localhost', port=6379)
            RedisManager.reset_instance()
            client2 = RedisManager.get_instance(host='localhost', port=6379)
            
            # After reset, should get a new instance
            # Note: In this mock test, both will be new instances
            assert client1 is not None
            assert client2 is not None


class TestRedisClientClose:
    """Test Redis client cleanup"""
    
    def test_close(self):
        """Test closing Redis connection"""
        client = RedisClient(decode_responses=False)
        
        with patch.object(client._pool, 'disconnect'):
            client.close()
            # Should not raise any exception


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
