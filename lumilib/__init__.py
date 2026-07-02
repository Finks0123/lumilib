# -*- coding: utf-8 -*-
# @Date  : 2026/6/22 16:56
# @Author: Finks
# @Desc  : lumilib 工具包导出模块

__version__ = '1.0.0'

from .common.logger import logger
from .common import file_utils
from .common import text_utils
from .common import decorators

from .dbm import (
    BaseDBManager,
    MongoDBManager,
    MongoDBConfig,
    PoolConfig,
    ReconnectConfig,
    FAISSManager,
    FAISSConfig,
    LMDBManager,
    LMDBConfig,
    RedisManager,
    RedisConfig,
    SQLiteManager,
    SQLiteConfig,
    ChromaManager,
    ChromaConfig,

)


def mongo(config=None):
    """创建 MongoDB 数据库管理器"""
    return MongoDBManager(config)


def faiss(config=None):
    """创建 FAISS 向量数据库管理器"""
    return FAISSManager(config)


def lmdb(config=None):
    """创建 LMDB 本地 KV 数据库管理器"""
    return LMDBManager(config)


def redis(config=None):
    """创建 Redis 数据库管理器"""
    return RedisManager(config)


def sqlite(config=None):
    """创建 SQLite 数据库管理器"""
    return SQLiteManager(config)


def chroma(config=None):
    """创建 Chroma 向量数据库管理器"""
    return ChromaManager(config)



__all__ = [
    '__version__',
    'logger',
    'file_utils',
    'text_utils',
    'decorators',
    'BaseDBManager',
    'MongoDBManager',
    'MongoDBConfig',
    'PoolConfig',
    'ReconnectConfig',
    'FAISSManager',
    'FAISSConfig',
    'LMDBManager',
    'LMDBConfig',
    'RedisManager',
    'RedisConfig',
    'SQLiteManager',
    'SQLiteConfig',
    'ChromaManager',
    'ChromaConfig',
    'mongo',
    'faiss',
    'lmdb',
    'redis',
    'sqlite',
    'chroma',
]