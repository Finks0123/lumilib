# -*- coding: utf-8 -*-
# @Date  : 2026/7/1
# @Author: Finks
# @Desc  : Redis 数据库封装（懒加载模式）

"""
Redis 数据库封装

特性：
    - 懒加载连接：只在首次访问时才建立连接
    - 线程安全：支持多线程并发访问
    - 自动重连：连接断开时自动尝试重连
    - 连接池：复用连接，提高性能
    - 支持主从：可选配置主从连接

使用示例：
    # 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import RedisManager, RedisConfig
    config = RedisConfig(host='localhost', port=6379)
    db = RedisManager(config)

    # 使用配置字典（兼容旧版）
    from lumilib.dbm import RedisManager
    db = RedisManager({'host': 'localhost', 'port': 6379})

    # 基本操作
    db.set('key1', 'value1')
    db.get('key1')
"""

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

try:
    import redis
except ImportError:
    raise ImportError("Redis is not installed. Please install redis.")

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager


class RedisConfig(BaseModel):
    host: str = Field(default='localhost', description="Redis 主机地址")
    port: int = Field(default=6379, description="Redis 端口")
    password: Optional[str] = Field(default=None, description="Redis 密码")
    db: int = Field(default=0, description="数据库编号")
    socket_timeout: Optional[float] = Field(default=None, description="连接超时时间(秒)")
    socket_connect_timeout: Optional[float] = Field(default=None, description="Socket 连接超时时间(秒)")
    max_connections: int = Field(default=10, description="连接池最大连接数")
    decode_responses: bool = Field(default=True, description="是否自动解码响应")
    health_check_interval: int = Field(default=20, description="健康检查间隔(秒)")
    retry_on_timeout: bool = Field(default=True, description="超时是否重试")


_help_doc = """
=== Redis 配置帮助 ===

【基础配置】
  host       : Redis 主机地址（默认: localhost）
               示例: host=redis.example.com

  port       : Redis 端口（默认: 6379）

  password   : Redis 密码（默认: None）
               如果 Redis 设置了密码，必须提供

  db         : 数据库编号（默认: 0）
               Redis 支持多个数据库，编号从 0 开始

【高级配置】
  socket_timeout       : 连接超时时间，秒（默认: None）
                        示例: socket_timeout=10.0

  socket_connect_timeout : Socket 连接超时时间，秒（默认: None）

  max_connections      : 连接池最大连接数（默认: 10）
                        根据并发需求调整

  decode_responses     : 是否自动解码响应（默认: True）
                        设置为 True 时，返回字符串而非字节

  health_check_interval: 健康检查间隔，秒（默认: 20）

  retry_on_timeout     : 超时是否重试（默认: True）

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import RedisManager, RedisConfig
  config = RedisConfig(host='localhost', port=6379, password='mypassword')
  db = RedisManager(config)

  # 方式2：使用字典配置（兼容旧版）
  db = RedisManager({'host': 'localhost', 'port': 6379})

  # 字符串操作
  db.set('key', 'value')
  result = db.get('key')
  db.delete('key')
  db.set_with_expire('key', 'value', ex=3600)

  # 哈希操作
  db.hset('user:1', {'name': 'John', 'age': '30'})
  db.hget('user:1', 'name')
  db.hgetall('user:1')

  # 列表操作
  db.lpush('queue', 'item1', 'item2')
  db.rpop('queue')

  # 集合操作
  db.sadd('users', 'user1', 'user2')
  db.smembers('users')

  # 管道操作
  with db.pipeline() as pipe:
      pipe.set('key1', 'value1')
      pipe.set('key2', 'value2')
      pipe.execute()

  # 获取底层连接
  r = db.connection
  r.set('raw_key', 'raw_value')
"""


class RedisManager(BaseDBManager):
    """
    Redis 数据库管理器

    该类封装了 Redis 数据库的核心功能，提供统一的接口用于：
    - 字符串操作：set, get, delete, expire
    - 哈希操作：hset, hget, hgetall
    - 列表操作：lpush, rpush, lpop, rpop
    - 集合操作：sadd, smembers, srem
    - 管道操作：批量执行命令
    - 连接池管理：复用连接
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'host': 'localhost',
        'port': 6379,
        'password': None,
        'db': 0,
        'socket_timeout': None,
        'socket_connect_timeout': None,
        'max_connections': 10,
        'decode_responses': True,
        'health_check_interval': 20,
        'retry_on_timeout': True,
    }

    def __init__(self, config=None):
        """
        初始化 Redis 管理器（懒加载模式）

        Args:
            config: 配置，支持：
                - RedisConfig Pydantic 模型（推荐）
                - 配置字典
        """
        self._client: Optional[redis.Redis] = None
        self._lock = threading.RLock()
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return RedisManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def config_help() -> str:
        """获取配置帮助信息"""
        print(_help_doc)
        return _help_doc

    def _connect(self) -> bool:
        """建立 Redis 连接（懒加载核心方法）"""
        host = self.get_config('host', 'localhost')
        port = self.get_config('port', 6379)
        password = self.get_config('password', None)
        db = self.get_config('db', 0)
        socket_timeout = self.get_config('socket_timeout', None)
        socket_connect_timeout = self.get_config('socket_connect_timeout', None)
        max_connections = self.get_config('max_connections', 10)
        decode_responses = self.get_config('decode_responses', True)
        health_check_interval = self.get_config('health_check_interval', 20)
        retry_on_timeout = self.get_config('retry_on_timeout', True)

        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                max_connections=max_connections,
                decode_responses=decode_responses,
                health_check_interval=health_check_interval,
                retry_on_timeout=retry_on_timeout,
            )
            self._client.ping()
            self._is_connected = True
            logger.info(f"Redis 连接成功: {host}:{port}/{db}")
            return True
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    def _reconnect_if_needed(self) -> bool:
        """检查并重新连接"""
        if self._client is None:
            return self._connect()
        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis 连接断开，尝试重连: {e}")
            return self._connect()

    def _close_connection(self):
        """关闭数据库连接"""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                    logger.info("Redis 连接已关闭")
                except Exception as e:
                    logger.warning(f"关闭 Redis 连接失败: {e}")
                finally:
                    self._client = None
                    self._is_connected = False

    @property
    def connection(self) -> redis.Redis:
        """获取 Redis 客户端实例"""
        self._ensure_connected()
        if self._client is None:
            raise RuntimeError("Redis 连接未建立")
        return self._client

    def set(self, key: str, value: Any, ex: Optional[int] = None, px: Optional[int] = None,
            nx: bool = False, xx: bool = False):
        """
        设置键值对

        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
            px: 过期时间（毫秒）
            nx: 仅当键不存在时设置
            xx: 仅当键存在时设置
        """
        with self._lock:
            self.connection.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
            logger.debug(f"设置键值对: {key}")

    def set_with_expire(self, key: str, value: Any, expire_seconds: int):
        """设置键值对并指定过期时间（秒）"""
        self.set(key, value, ex=expire_seconds)

    def get(self, key: str) -> Any:
        """获取键对应的值"""
        with self._lock:
            return self.connection.get(key)

    def delete(self, *keys: str) -> int:
        """删除键"""
        with self._lock:
            result = self.connection.delete(*keys)
            logger.debug(f"删除键: {keys}, 影响行数: {result}")
            return result

    def exists(self, *keys: str) -> int:
        """检查键是否存在"""
        with self._lock:
            return self.connection.exists(*keys)

    def expire(self, key: str, seconds: int) -> bool:
        """设置键的过期时间（秒）"""
        with self._lock:
            return self.connection.expire(key, seconds)

    def ttl(self, key: str) -> int:
        """获取键的剩余过期时间（秒）"""
        with self._lock:
            return self.connection.ttl(key)

    def incr(self, key: str, amount: int = 1) -> int:
        """自增键的值"""
        with self._lock:
            return self.connection.incr(key, amount)

    def decr(self, key: str, amount: int = 1) -> int:
        """自减键的值"""
        with self._lock:
            return self.connection.decr(key, amount)

    def hset(self, name: str, mapping: Optional[Dict[str, Any]] = None, **kwargs):
        """设置哈希字段"""
        with self._lock:
            result = self.connection.hset(name, mapping=mapping, **kwargs)
            logger.debug(f"设置哈希: {name}")
            return result

    def hget(self, name: str, key: str) -> Any:
        """获取哈希字段的值"""
        with self._lock:
            return self.connection.hget(name, key)

    def hgetall(self, name: str) -> Dict[str, Any]:
        """获取哈希的所有字段和值"""
        with self._lock:
            return self.connection.hgetall(name)

    def hkeys(self, name: str) -> List[str]:
        """获取哈希的所有字段名"""
        with self._lock:
            return self.connection.hkeys(name)

    def hvals(self, name: str) -> List[Any]:
        """获取哈希的所有值"""
        with self._lock:
            return self.connection.hvals(name)

    def hlen(self, name: str) -> int:
        """获取哈希的字段数量"""
        with self._lock:
            return self.connection.hlen(name)

    def hdel(self, name: str, *keys: str) -> int:
        """删除哈希字段"""
        with self._lock:
            result = self.connection.hdel(name, *keys)
            logger.debug(f"删除哈希字段: {name}.{keys}")
            return result

    def lpush(self, name: str, *values: Any) -> int:
        """从列表左侧添加元素"""
        with self._lock:
            result = self.connection.lpush(name, *values)
            logger.debug(f"左推列表: {name}")
            return result

    def rpush(self, name: str, *values: Any) -> int:
        """从列表右侧添加元素"""
        with self._lock:
            result = self.connection.rpush(name, *values)
            logger.debug(f"右推列表: {name}")
            return result

    def lpop(self, name: str) -> Any:
        """从列表左侧弹出元素"""
        with self._lock:
            return self.connection.lpop(name)

    def rpop(self, name: str) -> Any:
        """从列表右侧弹出元素"""
        with self._lock:
            return self.connection.rpop(name)

    def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """获取列表指定范围的元素"""
        with self._lock:
            return self.connection.lrange(name, start, end)

    def llen(self, name: str) -> int:
        """获取列表长度"""
        with self._lock:
            return self.connection.llen(name)

    def sadd(self, name: str, *values: Any) -> int:
        """向集合添加元素"""
        with self._lock:
            result = self.connection.sadd(name, *values)
            logger.debug(f"添加集合元素: {name}")
            return result

    def smembers(self, name: str) -> set:
        """获取集合的所有元素"""
        with self._lock:
            return self.connection.smembers(name)

    def srem(self, name: str, *values: Any) -> int:
        """从集合移除元素"""
        with self._lock:
            result = self.connection.srem(name, *values)
            logger.debug(f"移除集合元素: {name}")
            return result

    def scard(self, name: str) -> int:
        """获取集合的元素数量"""
        with self._lock:
            return self.connection.scard(name)

    def sismember(self, name: str, value: Any) -> bool:
        """检查元素是否在集合中"""
        with self._lock:
            return self.connection.sismember(name, value)

    def zadd(self, name: str, mapping: Optional[Dict[str, float]] = None, **kwargs):
        """向有序集合添加元素"""
        with self._lock:
            result = self.connection.zadd(name, mapping=mapping, **kwargs)
            logger.debug(f"添加有序集合元素: {name}")
            return result

    def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> Union[List[Any], List[Tuple[Any, float]]]:
        """获取有序集合指定范围的元素"""
        with self._lock:
            return self.connection.zrange(name, start, end, withscores=withscores)

    def zscore(self, name: str, value: str) -> Optional[float]:
        """获取有序集合元素的分数"""
        with self._lock:
            return self.connection.zscore(name, value)

    def zrem(self, name: str, *values: Any) -> int:
        """从有序集合移除元素"""
        with self._lock:
            result = self.connection.zrem(name, *values)
            logger.debug(f"移除有序集合元素: {name}")
            return result

    def keys(self, pattern: str = '*') -> List[str]:
        """获取匹配模式的键"""
        with self._lock:
            return self.connection.keys(pattern)

    def scan(self, cursor: int = 0, match: Optional[str] = None, count: int = 1000):
        """扫描键"""
        with self._lock:
            return self.connection.scan(cursor, match, count)

    def flushdb(self):
        """清空当前数据库"""
        with self._lock:
            self.connection.flushdb()
            logger.warning("Redis 数据库已清空")

    def flushall(self):
        """清空所有数据库"""
        with self._lock:
            self.connection.flushall()
            logger.warning("所有 Redis 数据库已清空")

    def pipeline(self, transaction: bool = True, shard_hint: Optional[str] = None):
        """获取管道对象"""
        return self.connection.pipeline(transaction=transaction, shard_hint=shard_hint)

    def close(self):
        """关闭数据库连接"""
        self._close_connection()

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = 'connected' if self.is_connected else 'disconnected'
        host = self.get_config('host', '?')
        port = self.get_config('port', '?')
        db = self.get_config('db', '?')
        return f"RedisManager(host={host}, port={port}, db={db}, status={status})"