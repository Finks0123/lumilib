# -*- coding: utf-8 -*-
# @Date  : 2026/7/1
# @Author: Finks
# @Desc  : Chroma 向量数据库封装（懒加载模式）

"""
Chroma 向量数据库封装

特性：
    - 懒加载连接：只在首次访问时才建立连接
    - 线程安全：支持多线程并发访问
    - 本地存储：支持本地文件系统存储
    - 向量搜索：支持相似度搜索
    - 元数据过滤：支持按元数据过滤搜索

使用示例：
    # 使用 Pydantic 配置模型（推荐）
    from lumilib.dbm import ChromaManager, ChromaConfig
    config = ChromaConfig(persist_directory='./chroma_data')
    db = ChromaManager(config)

    # 使用配置字典（兼容旧版）
    from lumilib.dbm import ChromaManager
    db = ChromaManager({'persist_directory': './chroma_data'})

    # 基本操作
    db.add(embeddings=vectors, documents=texts, ids=ids)
    results = db.search(query_vector, k=5)
"""

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("ChromaDB is not installed. Please install chromadb.")

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager


class ChromaConfig(BaseModel):
    persist_directory: str = Field(default='./chroma_data', description="持久化存储目录")
    collection_name: str = Field(default='default', description="集合名称")
    embedding_function: Optional[Any] = Field(default=None, description="自定义嵌入函数")
    allow_reset: bool = Field(default=True, description="是否允许重置集合")
    anonymized_telemetry: bool = Field(default=False, description="是否发送匿名遥测数据")


_help_doc = """
=== Chroma 配置帮助 ===

【基础配置】
  persist_directory : 持久化存储目录（默认: ./chroma_data）
                      示例: persist_directory=/data/chroma/myapp

  collection_name   : 集合名称（默认: default）
                      每个数据库可以包含多个集合

  embedding_function: 自定义嵌入函数（默认: None）
                      如果提供，将使用该函数生成向量
                      示例: embedding_function=my_embedding_function

【高级配置】
  allow_reset        : 是否允许重置集合（默认: True）
                      设置为 False 可防止意外清空数据

  anonymized_telemetry: 是否发送匿名遥测数据（默认: False）

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import ChromaManager, ChromaConfig
  config = ChromaConfig(persist_directory='./chroma_data')
  db = ChromaManager(config)

  # 方式2：使用字典配置（兼容旧版）
  db = ChromaManager({'persist_directory': './chroma_data'})

  # 添加向量和文档
  embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  documents = ["Hello world", "Goodbye world"]
  ids = ["id1", "id2"]
  db.add(embeddings=embeddings, documents=documents, ids=ids)

  # 搜索相似向量
  query_vector = [1.1, 2.1, 3.1]
  results = db.search(query_vector, k=2)

  # 添加元数据
  metadata = [{"category": "greeting"}, {"category": "farewell"}]
  db.add(embeddings=embeddings, documents=documents, ids=ids, metadata=metadata)

  # 按元数据过滤搜索
  results = db.search(query_vector, k=2, where={"category": "greeting"})

  # 更新数据
  db.update(ids=["id1"], embeddings=[[1.5, 2.5, 3.5]])

  # 删除数据
  db.delete(ids=["id2"])

  # 获取数据
  data = db.get(ids=["id1"])

  # 获取集合信息
  info = db.get_collection_info()

  # 切换集合
  db.set_collection('new_collection')

  # 获取底层集合对象
  collection = db.collection
"""


class ChromaManager(BaseDBManager):
    """
    Chroma 向量数据库管理器

    该类封装了 Chroma 向量数据库的核心功能，提供统一的接口用于：
    - 添加向量和文档
    - 向量搜索
    - 元数据过滤
    - 更新和删除数据
    - 集合管理
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'persist_directory': './chroma_data',
        'collection_name': 'default',
        'embedding_function': None,
        'allow_reset': True,
        'anonymized_telemetry': False,
    }

    def __init__(self, config=None):
        """
        初始化 Chroma 管理器（懒加载模式）

        Args:
            config: 配置，支持：
                - ChromaConfig Pydantic 模型（推荐）
                - 配置字典
        """
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[Any] = None
        self._lock = threading.RLock()
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return ChromaManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def config_help() -> str:
        """获取配置帮助信息"""
        print(_help_doc)
        return _help_doc

    def _connect(self) -> bool:
        """建立 Chroma 连接（懒加载核心方法）"""
        persist_directory = self.get_config('persist_directory', './chroma_data')
        embedding_function = self.get_config('embedding_function', None)
        allow_reset = self.get_config('allow_reset', True)
        anonymized_telemetry = self.get_config('anonymized_telemetry', False)

        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=anonymized_telemetry,
                ),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.get_config('collection_name', 'default'),
                embedding_function=embedding_function,
            )
            self._is_connected = True
            logger.info(f"Chroma 连接成功: {persist_directory}")
            return True
        except Exception as e:
            logger.error(f"Chroma 连接失败: {e}")
            raise

    def _reconnect_if_needed(self) -> bool:
        """检查并重新连接"""
        if self._client is None:
            return self._connect()
        return True

    def _close_connection(self):
        """关闭数据库连接"""
        with self._lock:
            if self._client is not None:
                try:
                    logger.info("Chroma 连接已关闭")
                except Exception as e:
                    logger.warning(f"关闭 Chroma 连接失败: {e}")
                finally:
                    self._client = None
                    self._collection = None
                    self._is_connected = False

    @property
    def connection(self) -> chromadb.Client:
        """获取 Chroma 客户端实例"""
        self._ensure_connected()
        if self._client is None:
            raise RuntimeError("Chroma 连接未建立")
        return self._client

    @property
    def collection(self):
        """获取当前集合实例"""
        self._ensure_connected()
        if self._collection is None:
            raise RuntimeError("Chroma 集合未初始化")
        return self._collection

    def set_collection(self, collection_name: str):
        """切换到指定集合"""
        with self._lock:
            self._collection = self.connection.get_or_create_collection(
                name=collection_name,
                embedding_function=self.get_config('embedding_function', None),
            )
            logger.info(f"切换到集合: {collection_name}")

    def add(self, embeddings: Optional[List[List[float]]] = None,
            documents: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        添加向量和文档

        Args:
            embeddings: 向量列表
            documents: 文档列表
            ids: ID 列表
            metadatas: 元数据列表
        """
        with self._lock:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
            logger.debug(f"添加 {len(ids) if ids else 0} 条数据")

    def add_with_text(self, embeddings: List[List[float]], texts: List[str],
                      ids: Optional[List[str]] = None):
        """
        添加向量和文本（便捷方法）

        Args:
            embeddings: 向量列表
            texts: 文本列表
            ids: ID 列表
        """
        self.add(embeddings=embeddings, documents=texts, ids=ids)

    def search(self, query: Union[List[float], List[List[float]]], k: int = 5,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, str]] = None,
               include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        搜索相似向量

        Args:
            query: 查询向量或向量列表
            k: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的字段，如 ['documents', 'metadatas', 'distances']

        Returns:
            Dict: 搜索结果
        """
        with self._lock:
            if include is None:
                include = ['documents', 'metadatas', 'distances']

            results = self.collection.query(
                query_embeddings=query,
                n_results=k,
                where=where,
                where_document=where_document,
                include=include,
            )
            logger.debug(f"搜索完成，返回 {k} 条结果")
            return results

    def search_with_text(self, query: Union[List[float], List[List[float]]], k: int = 5,
                         where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        搜索并返回文本（便捷方法）

        Args:
            query: 查询向量
            k: 返回结果数量
            where: 元数据过滤条件

        Returns:
            Dict: 搜索结果，包含文档和距离
        """
        return self.search(query, k, where, include=['documents', 'distances'])

    def get(self, ids: Optional[List[str]] = None,
            where: Optional[Dict[str, Any]] = None,
            where_document: Optional[Dict[str, str]] = None,
            include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取数据

        Args:
            ids: ID 列表
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的字段

        Returns:
            Dict: 查询结果
        """
        with self._lock:
            if include is None:
                include = ['documents', 'metadatas', 'embeddings']

            results = self.collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                include=include,
            )
            return results

    def get_texts(self, ids: List[str]) -> List[str]:
        """
        获取指定 ID 的文本（便捷方法）

        Args:
            ids: ID 列表

        Returns:
            List[str]: 文本列表
        """
        results = self.get(ids=ids, include=['documents'])
        return results.get('documents', [])

    def update(self, ids: List[str],
               embeddings: Optional[List[List[float]]] = None,
               documents: Optional[List[str]] = None,
               metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        更新数据

        Args:
            ids: ID 列表
            embeddings: 新向量列表
            documents: 新文档列表
            metadatas: 新元数据列表
        """
        with self._lock:
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(f"更新 {len(ids)} 条数据")

    def delete(self, ids: Optional[List[str]] = None,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, str]] = None):
        """
        删除数据

        Args:
            ids: ID 列表
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
        """
        with self._lock:
            self.collection.delete(
                ids=ids,
                where=where,
                where_document=where_document,
            )
            logger.debug(f"删除数据")

    def count(self) -> int:
        """获取集合中的数据数量"""
        with self._lock:
            return self.collection.count()

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        with self._lock:
            return {
                'name': self.collection.name,
                'count': self.collection.count(),
                'metadata': self.collection.metadata,
            }

    def reset_collection(self):
        """重置当前集合（清空所有数据）"""
        with self._lock:
            if self.get_config('allow_reset', True):
                self._client.delete_collection(name=self.collection.name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection.name,
                    embedding_function=self.get_config('embedding_function', None),
                )
                logger.warning("Chroma 集合已重置")
            else:
                raise RuntimeError("不允许重置集合，请设置 allow_reset=True")

    def list_collections(self) -> List[str]:
        """列出所有集合名称"""
        with self._lock:
            collections = self._client.list_collections()
            return [c.name for c in collections]

    def delete_collection(self, collection_name: str):
        """删除指定集合"""
        with self._lock:
            self._client.delete_collection(name=collection_name)
            logger.info(f"删除集合: {collection_name}")

    def close(self):
        """关闭数据库连接"""
        self._close_connection()

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._client is not None and self._is_connected

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = 'connected' if self.is_connected() else 'disconnected'
        persist_dir = self.get_config('persist_directory', '?')
        collection = self.get_config('collection_name', '?')
        return f"ChromaManager(dir={persist_dir}, collection={collection}, status={status})"