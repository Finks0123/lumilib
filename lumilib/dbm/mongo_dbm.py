# -*- coding: utf-8 -*-
# @Date  : 2026/6/4 15:30
# @Author: Finks
# @Desc  : MongoDB 客户端封装（懒加载模式）

"""
MongoDB 客户端封装

"""
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager


class PoolConfig(BaseModel):
    maxPoolSize: int = Field(default=50, description="连接池最大连接数")
    minPoolSize: int = Field(default=5, description="连接池最小空闲连接")
    maxIdleTimeMS: int = Field(default=60000, description="连接空闲超时(ms)")
    waitQueueTimeoutMS: int = Field(default=5000, description="等待连接队列超时(ms)")


class ReconnectConfig(BaseModel):
    max_retries: int = Field(default=3, description="连接失败最大重试次数")
    retry_delay: float = Field(default=1.0, description="初始重试间隔(秒)")
    retry_delay_multiplier: float = Field(default=2.0, description="重试间隔倍增系数")


class MongoDBConfig(BaseModel):
    mongo_url: str = Field(..., description="MongoDB连接地址，如 mongodb://user:pass@127.0.0.1:27017/")
    db_name: str = Field(..., description="目标数据库名称")
    pool: PoolConfig = Field(default_factory=PoolConfig, description="连接池配置")
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig, description="自动重连配置")


_help_doc = """
=== MongoDB 配置帮助 ===

【必填配置】
  mongo_url : MongoDB 连接地址（必须配置）
              示例: mongodb://localhost:27017/
              示例: mongodb://user:pass@host:27017/

  db_name   : 数据库名称（必须配置）
              示例: myapp

【可选配置 - 连接池（默认值）】
  pool.maxPoolSize       : 最大连接数 (默认: 50)
  pool.minPoolSize       : 最小连接数 (默认: 5)
  pool.maxIdleTimeMS     : 连接空闲超时，毫秒 (默认: 60000)
  pool.waitQueueTimeoutMS: 等待队列超时，毫秒 (默认: 5000)

【可选配置 - 重连（默认值）】
  reconnect.max_retries              : 最大重试次数 (默认: 3)
  reconnect.retry_delay              : 重试间隔，秒 (默认: 1.0)
  reconnect.retry_delay_multiplier   : 重试间隔倍数 (默认: 2)

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import MongoDBManager, MongoDBConfig
  config = MongoDBConfig(mongo_url='mongodb://localhost:27017/', db_name='myapp')
  db = MongoDBManager(config)

  # 方式2：使用字典配置（兼容旧版）
  db = MongoDBManager({
      'mongo_url': 'mongodb://user:pass@host:27017/',
      'db_name': 'myapp',
  })

  # 方式3：深度定制（连接池、重连配置）
  from lumilib.dbm import MongoDBConfig, PoolConfig, ReconnectConfig
  config = MongoDBConfig(
      mongo_url='mongodb://localhost:27017/',
      db_name='myapp',
      pool=PoolConfig(maxPoolSize=100),
      reconnect=ReconnectConfig(max_retries=5),
  )
  db = MongoDBManager(config)

  # 获取集合（便捷函数）
  users = dbm.get_collection('users')
  # 或使用属性访问
  users = dbm.users

  # 操作集合
  users.insert_one({'name': 'test'})
  users.find_one({'name': 'test'})
"""


class MongoDBManager(BaseDBManager):
    """
    MongoDB 客户端封装（懒加载模式）

    使用方式：
        # 使用配置模型
        config = MongoDBConfig(mongo_url='...', db_name='...')
        db = MongoDBManager(config)
        db.get_collection('users')
    """

    DEFAULT_CONFIG: Dict[str, Any] = {}

    POOL_CONFIG = {
        'maxPoolSize': 50,
        'minPoolSize': 5,
        'maxIdleTimeMS': 60000,
        'waitQueueTimeoutMS': 5000,
    }

    RECONNECT_CONFIG = {
        'max_retries': 3,
        'retry_delay': 1.0,
        'retry_delay_multiplier': 2,
    }

    def __init__(self, config=None):
        """
        初始化 MongoDB 管理器（懒加载，不立即连接）

        Args:
            config: 配置，支持：
                - MongoDBConfig Pydantic 模型（推荐）
                - 配置字典，支持的键：
                    - mongo_url: MongoDB 连接地址（必填）
                    - db_name: 数据库名称（必填）
                    - pool.*: 连接池配置（可选）
                    - reconnect.*: 重连配置（可选）
        """
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._last_health_check: float = 0
        self._health_check_interval: float = 30.0
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'pool': MongoDBManager.POOL_CONFIG.copy(),
            'reconnect': MongoDBManager.RECONNECT_CONFIG.copy(),
        }

    @staticmethod
    def config_help() -> str:
        """获取配置帮助信息"""
        print(_help_doc)
        return _help_doc

    def _build_connection_args(self) -> dict:
        """构建连接参数（合并连接池配置）"""
        args = {
            'serverSelectionTimeoutMS': 5000,
            'connectTimeoutMS': 5000,
            'socketTimeoutMS': 30000,
            'retryWrites': True,
            'retryReads': True,
        }
        pool_config = self.get_config('pool', {})
        if isinstance(pool_config, dict):
            args.update(pool_config)
        else:
            args.update(self.POOL_CONFIG)
        return args

    def _connect(self) -> bool:
        """建立 MongoDB 连接（懒加载核心方法）"""
        if self._client is not None:
            self._close_connection()

        retry_config = self.get_config('reconnect', self.RECONNECT_CONFIG)
        if not isinstance(retry_config, dict):
            retry_config = self.RECONNECT_CONFIG

        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 1.0)
        mongo_url = self.get_config('mongo_url')
        db_name = self.get_config('db_name')

        if not mongo_url:
            raise ValueError("mongo_url 未配置，请使用 set_config() 配置")

        if not db_name:
            raise ValueError("db_name 未配置，请使用 set_config() 配置")

        for attempt in range(1, int(max_retries + 1)):
            try:
                logger.info(f"正在连接 MongoDB... (尝试 {attempt}/{max_retries})")

                conn_args = self._build_connection_args()
                self._client = MongoClient(mongo_url, **conn_args)

                self._client.admin.command('ping')
                self._db = self._client[db_name]
                self._is_connected = True

                logger.info(f"MongoDB 连接成功，数据库: {db_name}")
                return True

            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"MongoDB 连接失败 (尝试 {attempt}/{max_retries}): {e}")

                if self._client is not None:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = None

                if attempt == max_retries:
                    logger.error(f"MongoDB 连接失败，已达最大重试次数")
                    raise ConnectionFailure(f"MongoDB 连接失败: {e}")

                delay = retry_delay * (retry_config.get('retry_delay_multiplier', 2) ** (attempt - 1))
                logger.info(f"等待 {delay} 秒后重试...")
                time.sleep(delay)

        return False

    def _close_connection(self):
        """关闭当前连接"""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("MongoDB 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 MongoDB 连接时出错: {e}")
            finally:
                self._client = None
                self._db = None
                self._is_connected = False

    def _check_health(self) -> bool:
        """健康检查"""
        current_time = time.time()

        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_connected

        self._last_health_check = current_time

        try:
            if self._client is not None:
                self._client.admin.command('ping')
                return True
        except Exception as e:
            logger.warning(f"MongoDB 健康检查失败: {e}")
            return False

        return False

    def _reconnect_if_needed(self) -> bool:
        """如果连接断开，尝试重新连接"""
        if self._client is None:
            return self._connect()

        if not self._check_health():
            logger.warning("MongoDB 连接不健康，尝试重新连接...")
            self._close_connection()
            return self._connect()

        return True

    @property
    def client(self) -> MongoClient:
        """获取 MongoClient 实例（懒加载）"""
        self._ensure_connected()
        if self._client is None:
            raise ConnectionFailure("MongoDB 连接不可用")
        return self._client

    @property
    def db(self) -> Database:
        """获取 Database 实例（懒加载）"""
        self._ensure_connected()
        if self._db is None:
            raise ConnectionFailure("MongoDB 数据库连接不可用")
        return self._db

    @property
    def mongo_url(self) -> str:
        """获取 mongo_url"""
        return self.get_config('mongo_url', '')

    @property
    def db_name(self) -> str:
        """获取 db_name"""
        return self.get_config('db_name', '')

    def close(self):
        """关闭连接"""
        self._close_connection()

    def is_connected(self) -> bool:
        """检查是否已连接"""
        if self._client is None:
            return False
        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            return False

    def get_pool_stats(self) -> dict:
        """获取连接池状态"""
        if self._client is None:
            return {'status': 'disconnected'}

        try:
            server_info = self._client.server_info()
            pool_config = self.get_config('pool', self.POOL_CONFIG)
            return {
                'status': 'connected',
                'server_version': server_info.get('version', 'unknown'),
                'pool_size': pool_config.get('maxPoolSize', 'unknown'),
                'database': self.get_config('db_name'),
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_collection(self, collection_name: str):
        """
        获取指定名称的集合（懒加载）

        Args:
            collection_name: 集合名称

        Returns:
            pymongo.collection.Collection: 集合对象
        """
        return self.db[collection_name]

    def __getattr__(self, name: str):
        """
        动态属性访问，实现懒加载

        例如：db.users 相当于 db.get_collection('users')
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in ('client', 'db', 'mongo_url', 'db_name', 'config', 'is_connected'):
            return object.__getattribute__(self, name)

        return self.get_collection(name)

    def __repr__(self) -> str:
        status = 'connected' if self.is_connected() else 'disconnected'
        return f"MongoDBManager(db={self.get_config('db_name', '?')}, status={status})"