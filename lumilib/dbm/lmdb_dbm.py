# -*- coding: utf-8 -*-
# @Date  : 2026/7/1
# @Author: Finks
# @Desc  : LMDB 本地 KV 数据库封装（懒加载模式）

"""
LMDB 本地 KV 数据库封装

特性：
    - 懒加载连接：只在首次访问时才打开数据库
    - 高性能：基于内存映射，读写速度极快
    - 事务支持：ACID 事务保证
    - 线程安全：支持多线程并发读写
    - 自动持久化：数据写入后自动落盘

使用示例：
    # 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import LMDBManager, LMDBConfig
    config = LMDBConfig(db_path='./lmdb_data')
    db = LMDBManager(config)

    # 使用配置字典（兼容旧版）
    from lumilib.dbm import LMDBManager
    db = LMDBManager({'db_path': './lmdb_data'})

    # 基本操作
    db.put('key1', 'value1')
    db.get('key1')
"""

import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    import lmdb
except ImportError:
    raise ImportError("LMDB is not installed. Please install lmdb.")

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager


class LMDBConfig(BaseModel):
    db_path: str = Field(default='./lmdb_data', description="数据库存储目录")
    map_size: int = Field(default=1073741824, description="数据库最大容量，字节（默认: 1GB）")
    max_dbs: int = Field(default=1, description="最大数据库数量，允许创建多个命名数据库")
    readonly: bool = Field(default=False, description="是否以只读模式打开")
    lock: bool = Field(default=True, description="是否启用锁，设为 False 可提高性能，但不支持并发写入")
    mode: int = Field(default=0o755, description="文件权限")


_help_doc = """
=== LMDB 配置帮助 ===

【基础配置】
  db_path   : 数据库存储目录（默认: ./lmdb_data）
              示例: db_path=/data/lmdb/myapp

  map_size  : 数据库最大容量，字节（默认: 1073741824，即 1GB）
              示例: map_size=2147483648  # 2GB

【高级配置】
  max_dbs   : 最大数据库数量（默认: 1）
              允许创建多个命名数据库

  readonly  : 是否以只读模式打开（默认: False）

  lock      : 是否启用锁（默认: True）
              设为 False 可提高性能，但不支持并发写入

  mode      : 文件权限（默认: 0o755）

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import LMDBManager, LMDBConfig
  config = LMDBConfig(db_path='./lmdb_data', map_size=2147483648)
  db = LMDBManager(config)

  # 方式2：使用字典配置（兼容旧版）
  db = LMDBManager({'db_path': './lmdb_data'})

  # 基础操作
  db.put('key', 'value')
  result = db.get('key')

  # 使用事务
  with db.transaction() as txn:
      txn.put(b'key1', b'value1')
      txn.put(b'key2', b'value2')

  # 批量操作
  db.put_batch({'k1': 'v1', 'k2': 'v2'})

  # 遍历所有键值
  for key, value in db.iter_items():
      print(key, value)

  # 使用多数据库
  config = LMDBConfig(db_path='./lmdb_data', max_dbs=10)
  db = LMDBManager(config)
  db.open_db('users')
  db.put('user:1', 'data', db_name='users')
"""


class LMDBManager(BaseDBManager):
    """
    LMDB 本地 KV 数据库管理器

    该类封装了 LMDB 数据库的核心功能，提供统一的接口用于：
    - 键值对的增删改查
    - 批量操作
    - 事务支持
    - 多数据库管理
    - 线程安全的并发访问
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'db_path': './lmdb_data',
        'map_size': 1073741824,
        'max_dbs': 1,
        'readonly': False,
        'lock': True,
        'mode': 0o755,
    }

    def __init__(self, config=None):
        """
        初始化 LMDB 管理器（懒加载模式）

        Args:
            config: 配置，支持：
                - LMDBConfig Pydantic 模型（推荐）
                - 配置字典
        """
        self._env: Optional[lmdb.Environment] = None
        self._dbs: Dict[str, lmdb.Database] = {}
        self._lock = threading.RLock()
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return LMDBManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def config_help() -> str:
        """获取配置帮助信息"""
        print(_help_doc)
        return _help_doc

    def _connect(self) -> bool:
        """打开 LMDB 数据库（懒加载核心方法）"""
        db_path = self.get_config('db_path', './lmdb_data')
        map_size = self.get_config('map_size', 1073741824)
        max_dbs = self.get_config('max_dbs', 1)
        readonly = self.get_config('readonly', False)
        lock = self.get_config('lock', True)
        mode = self.get_config('mode', 0o755)

        os.makedirs(db_path, exist_ok=True)

        try:
            self._env = lmdb.open(
                db_path,
                map_size=map_size,
                max_dbs=max_dbs,
                readonly=readonly,
                lock=lock,
                mode=mode,
                subdir=True,
            )
            self._is_connected = True
            logger.info(f"LMDB 数据库打开成功: {db_path}")
            return True
        except Exception as e:
            logger.error(f"LMDB 数据库打开失败: {e}")
            raise

    def _reconnect_if_needed(self) -> bool:
        """检查并重新连接"""
        if self._env is None:
            return self._connect()
        return True

    def _close_connection(self):
        """关闭数据库连接"""
        with self._lock:
            for db_name, db in self._dbs.items():
                try:
                    db.close()
                except Exception as e:
                    logger.warning(f"关闭数据库 {db_name} 失败: {e}")
            self._dbs.clear()

            if self._env is not None:
                try:
                    self._env.close()
                    logger.info("LMDB 数据库已关闭")
                except Exception as e:
                    logger.warning(f"关闭 LMDB 环境失败: {e}")
                finally:
                    self._env = None
                    self._is_connected = False

    @property
    def env(self) -> lmdb.Environment:
        """获取 LMDB 环境实例"""
        self._ensure_connected()
        if self._env is None:
            raise RuntimeError("LMDB 数据库未打开")
        return self._env

    def open_db(self, name: str):
        """打开或创建一个命名数据库"""
        with self._lock:
            if name in self._dbs:
                return self._dbs[name]

            db = self.env.open_db(name.encode())
            self._dbs[name] = db
            logger.info(f"打开数据库: {name}")
            return db

    def transaction(self, write: bool = True):
        """获取事务对象"""
        return self.env.begin(write=write)

    def put(self, key: str, value: str, db_name: str = None):
        """写入键值对"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), value.encode(), db=db)
            logger.debug(f"写入键值对: {key}")

    def get(self, key: str, db_name: str = None) -> Optional[str]:
        """获取键对应的值"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode(), db=db)
            if value is not None:
                return value.decode()
            return None

    def delete(self, key: str, db_name: str = None):
        """删除键值对"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=True) as txn:
                txn.delete(key.encode(), db=db)
            logger.debug(f"删除键值对: {key}")

    def put_batch(self, data: Dict[str, str], db_name: str = None):
        """批量写入键值对"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=True) as txn:
                for key, value in data.items():
                    txn.put(key.encode(), value.encode(), db=db)
            logger.debug(f"批量写入 {len(data)} 个键值对")

    def get_batch(self, keys: List[str], db_name: str = None) -> Dict[str, str]:
        """批量获取键值对"""
        result = {}
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=False) as txn:
                for key in keys:
                    value = txn.get(key.encode(), db=db)
                    if value is not None:
                        result[key] = value.decode()
        return result

    def delete_batch(self, keys: List[str], db_name: str = None):
        """批量删除键值对"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=True) as txn:
                for key in keys:
                    txn.delete(key.encode(), db=db)
            logger.debug(f"批量删除 {len(keys)} 个键值对")

    def iter_items(self, db_name: str = None) -> Iterator[Tuple[str, str]]:
        """遍历所有键值对"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=False) as txn:
                cursor = txn.cursor(db=db)
                for key, value in cursor:
                    yield key.decode(), value.decode()

    def iter_keys(self, db_name: str = None) -> Iterator[str]:
        """遍历所有键"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=False) as txn:
                cursor = txn.cursor(db=db)
                for key, _ in cursor:
                    yield key.decode()

    def contains(self, key: str, db_name: str = None) -> bool:
        """检查键是否存在"""
        return self.get(key, db_name) is not None

    def get_stats(self, db_name: str = None) -> dict:
        """获取数据库统计信息"""
        with self._lock:
            db = self._dbs.get(db_name) if db_name else None
            with self.env.begin(write=False) as txn:
                stats = txn.stat(db=db)
                return {
                    'entries': stats.get('entries', 0),
                    'branch_pages': stats.get('branch_pages', 0),
                    'leaf_pages': stats.get('leaf_pages', 0),
                    'overflow_pages': stats.get('overflow_pages', 0),
                }

    def sync(self, force: bool = True):
        """强制同步到磁盘"""
        with self._lock:
            self.env.sync(force=force)
            logger.debug("LMDB 已同步到磁盘")

    def close(self):
        """关闭数据库连接"""
        self._close_connection()

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._env is not None and self._is_connected

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = 'connected' if self.is_connected else 'disconnected'
        db_path = self.get_config('db_path', '?')
        return f"LMDBManager(path={db_path}, status={status})"