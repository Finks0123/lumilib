# -*- coding: utf-8 -*-
# @Date  : 2026/6/29
# @Author: Finks
# @Desc  : 数据库管理模块

"""
数据库管理模块 - 懒加载、零配置、优雅连接

特性：
    - 懒加载连接：只在首次访问数据库时才建立连接
    - Pydantic 配置：类型安全的配置模型，自动提示和验证
    - 默认配置：内置合理默认值，开箱即用
    - 深度定制：支持细粒度配置
    - 2行代码：极简使用方式

【快速开始】

    # MongoDB - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import MongoDBManager, MongoDBConfig
    config = MongoDBConfig(mongo_url='mongodb://localhost:27017/', db_name='myapp')
    db = MongoDBManager(config)
    db.users.find_one()  # 首次访问时自动连接

    # FAISS - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import FAISSManager, FAISSConfig
    config = FAISSConfig(dimension=128, index_type='flat')
    db = FAISSManager(config)
    db.add(vectors)
    db.search(query, k=5)

    # LMDB - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import LMDBManager, LMDBConfig
    config = LMDBConfig(db_path='./lmdb_data')
    db = LMDBManager(config)
    db.put('key', 'value')

    # Redis - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import RedisManager, RedisConfig
    config = RedisConfig(host='localhost', port=6379)
    db = RedisManager(config)
    db.set('key', 'value')

    # SQLite - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import SQLiteManager, SQLiteConfig
    config = SQLiteConfig(db_path='./database.db')
    db = SQLiteManager(config)
    db.execute('CREATE TABLE users (id INTEGER PRIMARY KEY)')

    # Chroma - 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import ChromaManager, ChromaConfig
    config = ChromaConfig(persist_directory='./chroma_data')
    db = ChromaManager(config)
    db.add(embeddings=vectors, documents=texts)

【配置方式】

    1. Pydantic 配置模型（推荐）
       from lumilib.dbm import MongoDBManager, MongoDBConfig, PoolConfig
       config = MongoDBConfig(
           mongo_url='mongodb://localhost:27017/',
           db_name='myapp',
           pool=PoolConfig(maxPoolSize=100),
       )
       db = MongoDBManager(config)

    2. 字典配置（兼容旧版）
       from lumilib.dbm import MongoDBManager
       db = MongoDBManager({
           'mongo_url': 'mongodb://user:pass@host:27017/',
           'db_name': 'myapp',
       })

    3. 查看配置帮助
       from lumilib.dbm import db_help
       db_help()
"""

from lumilib.dbm.base_db_manager import BaseDBManager
from lumilib.dbm.mongo_dbm import MongoDBManager, MongoDBConfig, PoolConfig, ReconnectConfig
from lumilib.dbm.faiss_dbm import FAISSManager, FAISSConfig
from lumilib.dbm.lmdb_dbm import LMDBManager, LMDBConfig
from lumilib.dbm.redis_dbm import RedisManager, RedisConfig
from lumilib.dbm.sqlite_dbm import SQLiteManager, SQLiteConfig
from lumilib.dbm.chroma_dbm import ChromaManager, ChromaConfig




__all__ = [
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
]