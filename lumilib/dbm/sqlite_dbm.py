# -*- coding: utf-8 -*-
# @Date  : 2026/7/1
# @Author: Finks
# @Desc  : SQLite 数据库封装（懒加载模式）

"""
SQLite 数据库封装

特性：
    - 懒加载连接：只在首次访问时才打开数据库
    - 线程安全：支持多线程并发访问
    - 事务支持：ACID 事务保证
    - 自动提交：可配置自动提交模式
    - 连接池：复用连接，提高性能

使用示例：
    # 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import SQLiteManager, SQLiteConfig
    config = SQLiteConfig(db_path='./example.db')
    db = SQLiteManager(config)

    # 使用配置字典（兼容旧版）
    from lumilib.dbm import SQLiteManager
    db = SQLiteManager({'db_path': './example.db'})

    # 基本操作
    db.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')
    db.insert('users', {'name': 'John'})
    result = db.query('SELECT * FROM users')
"""

import os
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager


class SQLiteConfig(BaseModel):
    db_path: str = Field(default='./database.db', description="数据库文件路径")
    timeout: float = Field(default=5.0, description="连接超时时间(秒)")
    detect_types: int = Field(default=0, description="类型检测标志")
    isolation_level: Optional[str] = Field(default=None, description="事务隔离级别")
    check_same_thread: bool = Field(default=True, description="是否检查同一线程")
    max_connections: int = Field(default=5, description="连接池最大连接数")


_help_doc = """
=== SQLite 配置帮助 ===

【基础配置】
  db_path   : 数据库文件路径（默认: ./database.db）
              示例: db_path=/data/sqlite/myapp.db
              特殊值: ':memory:' 表示内存数据库

【高级配置】
  timeout       : 连接超时时间，秒（默认: 5.0）
                  当多个连接同时访问数据库时，超时等待时间

  detect_types  : 类型检测标志（默认: 0）
                  可选值:
                    - 0: 不检测类型
                    - sqlite3.PARSE_DECLTYPES: 解析声明类型
                    - sqlite3.PARSE_COLNAMES: 解析列名

  isolation_level: 事务隔离级别（默认: None）
                   可选值:
                     - None: 默认隔离级别（可序列化）
                     - 'DEFERRED': 延迟锁定
                     - 'IMMEDIATE': 立即锁定
                     - 'EXCLUSIVE': 独占锁定

  check_same_thread: 是否检查同一线程（默认: True）
                     设置为 False 时，允许在多线程间共享连接

  max_connections : 连接池最大连接数（默认: 5）

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import SQLiteManager, SQLiteConfig
  config = SQLiteConfig(db_path='./example.db')
  db = SQLiteManager(config)

  # 方式2：使用字典配置（兼容旧版）
  db = SQLiteManager({'db_path': './example.db'})

  # 创建表
  db.execute('''
      CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT UNIQUE,
          age INTEGER,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
  ''')

  # 插入数据
  db.insert('users', {'name': 'John', 'email': 'john@example.com', 'age': 30})

  # 查询数据
  result = db.query('SELECT * FROM users WHERE age > ?', (25,))
  for row in result:
      print(row)

  # 更新数据
  db.update('users', {'age': 31}, 'id = ?', (1,))

  # 删除数据
  db.delete('users', 'id = ?', (1,))

  # 批量插入
  data = [
      {'name': 'Alice', 'email': 'alice@example.com'},
      {'name': 'Bob', 'email': 'bob@example.com'}
  ]
  db.insert_batch('users', data)

  # 事务操作
  with db.transaction() as conn:
      conn.execute('INSERT INTO users (name) VALUES (?)', ('Charlie',))
      conn.execute('INSERT INTO users (name) VALUES (?)', ('David',))

  # 使用连接池
  config = SQLiteConfig(db_path='./example.db', max_connections=10)
  db = SQLiteManager(config)

  # 获取底层连接
  conn = db.connection
  cursor = conn.cursor()
"""


class SQLiteManager(BaseDBManager):
    """
    SQLite 数据库管理器

    该类封装了 SQLite 数据库的核心功能，提供统一的接口用于：
    - 执行 SQL 语句
    - 查询数据
    - 插入、更新、删除数据
    - 批量操作
    - 事务管理
    - 连接池管理
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'db_path': './database.db',
        'timeout': 5.0,
        'detect_types': 0,
        'isolation_level': None,
        'check_same_thread': True,
        'max_connections': 5,
    }

    def __init__(self, config=None):
        """
        初始化 SQLite 管理器（懒加载模式）

        Args:
            config: 配置，支持：
                - SQLiteConfig Pydantic 模型（推荐）
                - 配置字典
        """
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return SQLiteManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def config_help() -> str:
        """获取配置帮助信息"""
        print(_help_doc)
        return _help_doc

    def _connect(self) -> bool:
        """打开 SQLite 数据库（懒加载核心方法）"""
        db_path = self.get_config('db_path', './database.db')
        timeout = self.get_config('timeout', 5.0)
        detect_types = self.get_config('detect_types', 0)
        isolation_level = self.get_config('isolation_level', None)
        check_same_thread = self.get_config('check_same_thread', True)

        if db_path != ':memory:':
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        try:
            self._conn = sqlite3.connect(
                db_path,
                timeout=timeout,
                detect_types=detect_types,
                isolation_level=isolation_level,
                check_same_thread=check_same_thread,
            )
            self._conn.row_factory = sqlite3.Row
            self._is_connected = True
            logger.info(f"SQLite 数据库打开成功: {db_path}")
            return True
        except Exception as e:
            logger.error(f"SQLite 数据库打开失败: {e}")
            raise

    def _reconnect_if_needed(self) -> bool:
        """检查并重新连接"""
        if self._conn is None:
            return self._connect()
        try:
            self._conn.execute('SELECT 1')
            return True
        except Exception as e:
            logger.warning(f"SQLite 连接断开，尝试重连: {e}")
            return self._connect()

    def _close_connection(self):
        """关闭数据库连接"""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    logger.info("SQLite 数据库已关闭")
                except Exception as e:
                    logger.warning(f"关闭 SQLite 连接失败: {e}")
                finally:
                    self._conn = None
                    self._is_connected = False

    @property
    def connection(self) -> sqlite3.Connection:
        """获取 SQLite 连接实例"""
        self._ensure_connected()
        if self._conn is None:
            raise RuntimeError("SQLite 连接未建立")
        return self._conn

    def execute(self, sql: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """
        执行 SQL 语句

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            sqlite3.Cursor: 游标对象
        """
        with self._lock:
            cursor = self.connection.cursor()
            try:
                if params:
                    result = cursor.execute(sql, params)
                else:
                    result = cursor.execute(sql)
                self.connection.commit()
                logger.debug(f"执行 SQL: {sql[:50]}...")
                return result
            except Exception as e:
                self.connection.rollback()
                logger.error(f"执行 SQL 失败: {e}")
                raise

    def execute_script(self, sql_script: str):
        """执行 SQL 脚本（多条语句）"""
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.executescript(sql_script)
                self.connection.commit()
                logger.debug(f"执行 SQL 脚本")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"执行 SQL 脚本失败: {e}")
                raise

    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        查询数据

        Args:
            sql: SELECT 语句
            params: 参数元组

        Returns:
            List[Dict]: 结果列表，每行是一个字典
        """
        with self._lock:
            cursor = self.connection.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"查询失败: {e}")
                raise

    def query_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """
        查询单条数据

        Args:
            sql: SELECT 语句
            params: 参数元组

        Returns:
            Dict or None: 结果字典或 None
        """
        with self._lock:
            cursor = self.connection.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
            except Exception as e:
                logger.error(f"查询失败: {e}")
                raise

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        插入单条数据

        Args:
            table: 表名
            data: 数据字典

        Returns:
            int: 插入的行 ID
        """
        with self._lock:
            keys = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = tuple(data.values())

            sql = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
            cursor = self.connection.cursor()
            try:
                cursor.execute(sql, values)
                self.connection.commit()
                logger.debug(f"插入数据: {table}, ID: {cursor.lastrowid}")
                return cursor.lastrowid
            except Exception as e:
                self.connection.rollback()
                logger.error(f"插入失败: {e}")
                raise

    def insert_batch(self, table: str, data_list: List[Dict[str, Any]]):
        """
        批量插入数据

        Args:
            table: 表名
            data_list: 数据字典列表
        """
        if not data_list:
            return

        with self._lock:
            keys = ', '.join(data_list[0].keys())
            placeholders = ', '.join(['?' for _ in data_list[0]])
            values_list = [tuple(d.values()) for d in data_list]

            sql = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
            cursor = self.connection.cursor()
            try:
                cursor.executemany(sql, values_list)
                self.connection.commit()
                logger.debug(f"批量插入 {len(data_list)} 条数据: {table}")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"批量插入失败: {e}")
                raise

    def update(self, table: str, data: Dict[str, Any], where_clause: str, params: Optional[Tuple] = None):
        """
        更新数据

        Args:
            table: 表名
            data: 更新的数据字典
            where_clause: WHERE 子句
            params: WHERE 子句参数
        """
        with self._lock:
            set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
            values = tuple(data.values())
            if params:
                values += params

            sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            cursor = self.connection.cursor()
            try:
                cursor.execute(sql, values)
                self.connection.commit()
                logger.debug(f"更新数据: {table}, 影响行数: {cursor.rowcount}")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"更新失败: {e}")
                raise

    def delete(self, table: str, where_clause: str, params: Optional[Tuple] = None):
        """
        删除数据

        Args:
            table: 表名
            where_clause: WHERE 子句
            params: WHERE 子句参数
        """
        with self._lock:
            sql = f"DELETE FROM {table} WHERE {where_clause}"
            cursor = self.connection.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                self.connection.commit()
                logger.debug(f"删除数据: {table}, 影响行数: {cursor.rowcount}")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"删除失败: {e}")
                raise

    def transaction(self):
        """获取事务上下文管理器"""
        return TransactionContext(self)

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        创建表

        Args:
            table_name: 表名
            columns: 列定义字典，如 {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT NOT NULL'}
        """
        column_defs = ', '.join([f"{name} {typ}" for name, typ in columns.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})"
        self.execute(sql)
        logger.info(f"创建表: {table_name}")

    def drop_table(self, table_name: str):
        """删除表"""
        sql = f"DROP TABLE IF EXISTS {table_name}"
        self.execute(sql)
        logger.info(f"删除表: {table_name}")

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        sql = '''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        '''
        result = self.query_one(sql, (table_name,))
        return result is not None

    def get_tables(self) -> List[str]:
        """获取所有表名"""
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        results = self.query(sql)
        return [row['name'] for row in results]

    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表的列信息"""
        sql = f"PRAGMA table_info({table_name})"
        results = self.query(sql)
        return results

    def vacuum(self):
        """执行 VACUUM 命令，优化数据库"""
        self.execute('VACUUM')
        logger.info("执行 VACUUM 命令")

    def close(self):
        """关闭数据库连接"""
        self._close_connection()

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        if self._conn is None:
            return False
        try:
            self._conn.execute('SELECT 1')
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = 'connected' if self.is_connected else 'disconnected'
        db_path = self.get_config('db_path', '?')
        return f"SQLiteManager(path={db_path}, status={status})"


class TransactionContext:
    """事务上下文管理器"""

    def __init__(self, manager: SQLiteManager):
        self._manager = manager
        self._conn = None

    def __enter__(self):
        """进入事务"""
        self._conn = self._manager.connection
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出事务"""
        if exc_type is None:
            try:
                self._conn.commit()
                logger.debug("事务提交成功")
            except Exception as e:
                self._conn.rollback()
                logger.error(f"事务提交失败: {e}")
                raise
        else:
            self._conn.rollback()
            logger.debug("事务回滚")