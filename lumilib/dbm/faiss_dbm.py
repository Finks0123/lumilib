# -*- coding: utf-8 -*-
# @Date  : 2026/6/29
# @Author: Finks
# @Desc  : FAISS 向量数据库封装（懒加载模式）
"""
FAISS 向量数据库封装
"""

import os
import pickle
import threading
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")

from lumilib.common.logger import logger
from lumilib.dbm.base_db_manager import BaseDBManager

_help_doc = """
=== FAISS 配置帮助 ===

【基础配置】
  dimension : 向量维度（必须与数据一致，默认: 128）
              示例: dimension=256

  index_type: 索引类型（默认: flat）
              可选值: flat, ivf, hnsw, pq, ivf_pq
              说明:
                flat   : 暴力搜索，精度最高，适合小数据集（万级）
                ivf    : 倒排索引，需要训练，适合中等数据集（百万级）
                hnsw   : 分层可导航小世界图，无需训练，高速搜索
                pq     : 产品量化，大幅压缩内存，需要训练
                ivf_pq : IVF + PQ 组合，适合超大数据集（十亿级）

  metric    : 距离度量（默认: l2）
              可选值: l2, ip, cosine
              说明:
                l2      : 欧氏距离，适用于大多数场景
                ip      : 内积，适用于向量已归一化的场景
                cosine  : 余弦相似度，内部使用内积并自动归一化

【索引参数】
  nlist     : IVF/IVF_PQ 索引的聚类中心数量（默认: 100）
              建议值: 数据集大小的平方根左右

  nprobe    : IVF/IVF_PQ 搜索时探测的聚类中心数量（默认: 10）
              值越大精度越高，但速度越慢

  m         : PQ/IVF_PQ 索引的子向量数量（默认: 16）
              要求: m * nbits <= 32，且 dimension 能被 m 整除

  nbits     : PQ/IVF_PQ 每个子向量的比特数（默认: 8）
              可选值: 4, 8, 12, 16

  M         : HNSW 索引每个节点的连接数（默认: 32）
              值越大搜索精度越高，但内存占用越大

  efSearch  : HNSW 搜索时的候选节点数（默认: 64）
              值越大精度越高，但搜索速度越慢

【高级配置】
  use_gpu   : 是否使用 GPU 加速（默认: False）
              需要安装 faiss-gpu 版本

  gpu_id    : GPU 设备 ID（默认: 0）
              当有多块 GPU 时指定使用哪一块

  num_threads: CPU 搜索线程数（默认: 0，使用系统默认）
              值越大搜索速度越快，但占用更多 CPU 资源

【存储配置】
  index_dir : 索引文件存储目录（默认: ./faiss）
              支持绝对路径和相对路径

  index_name: 索引文件名（默认: faiss.index）

  auto_persist: 是否自动持久化（默认: False）
                启用后，在添加或删除向量时会自动保存到文件

【使用示例】

  # 方式1：使用 Pydantic 配置模型（推荐）
  from lumilib.dbm import FAISSManager, FAISSConfig
  config = FAISSConfig(dimension=128, index_type='flat')
  db = FAISSManager(config)
  db.add(vectors)
  distances, indices = db.search(query, k=5)

  # 方式2：使用字典配置（兼容旧版）
  db = FAISSManager({'dimension': 128, 'index_type': 'flat'})
  db.add(vectors)
  distances, indices = db.search(query, k=5)

  # 使用自定义 ID（默认支持）
  db = FAISSManager({'dimension': 128})
  db.add(vectors, ids)
  db.remove(remove_ids)

  # 添加带元数据的向量
  db.add_with_metadata(vectors, metadata)
  texts = db.get_metadata(indices)

  # 使用自动持久化
  db = FAISSManager({'dimension': 128, 'auto_persist': True})
  db.add(vectors)  # 自动保存到默认路径

  # 使用 HNSW 高速索引
  db = FAISSManager({'dimension': 128, 'index_type': 'hnsw', 'M': 32})

  # 使用 IVF 索引
  db = FAISSManager({'dimension': 128, 'index_type': 'ivf', 'nlist': 100, 'nprobe': 20})

  # 保存和加载（包含文本索引）
  db.save()                        # 保存到默认路径
  db.save('my_index.faiss')        # 保存到指定路径
  db = FAISSManager.load('my_index.faiss')
"""

from pydantic import BaseModel, Field
from typing import Optional

class FAISSConfig(BaseModel):
    dimension: int = Field(default=128, description="向量维度，必须与数据一致")
    index_dir: str = Field(default='./faiss', description="索引文件存储目录")
    index_name: str = Field(default='faiss.index', description="索引文件名")
    index_type: str = Field(default='flat', description="索引类型：flat/ivf/hnsw/pq/ivf_pq")
    metric: str = Field(default='l2', description="距离度量：l2/ip/cosine")
    auto_persist: bool = Field(default=False, description="是否自动持久化")
    nlist: int = Field(default=100, description="IVF 聚类中心数量")
    nprobe: int = Field(default=10, description="IVF 搜索探测数量")
    m: int = Field(default=16, description="PQ 子向量数量")
    nbits: int = Field(default=8, description="PQ 比特数")
    M: int = Field(default=32, description="HNSW 节点连接数")
    efSearch: int = Field(default=64, description="HNSW 搜索参数")
    use_gpu: bool = Field(default=False, description="是否使用 GPU")
    gpu_id: int = Field(default=0, description="GPU 设备 ID")
    num_threads: int = Field(default=0, description="CPU 线程数，0 表示使用系统默认")


class FAISSManager(BaseDBManager):
    """
    FAISS 向量数据库管理器

    该类封装了 FAISS 向量数据库的核心功能，提供统一的接口用于：
    - 创建和管理多种类型的向量索引
    - 添加、删除、搜索向量
    - 保存和加载索引（包含文本索引字典）
    - GPU 加速和多线程搜索

    配置项说明：
        dimension:      向量维度，必须与数据一致
        index_type:     索引类型，决定搜索算法和性能特征
        metric:         距离度量方式，影响相似度计算
        index_dir:      索引文件存储目录，默认 ./faiss
        index_name:     索引文件名，默认 faiss.index
        use_id_map:     是否启用 ID 映射，启用后支持自定义 ID
        nlist:          IVF 索引的聚类中心数量
        nprobe:         IVF 搜索时探测的聚类中心数量
        m:              PQ 索引的子向量数量
        nbits:          PQ 每个子向量的比特数
        M:              HNSW 索引每个节点的连接数
        efSearch:       HNSW 搜索时的候选节点数
        use_gpu:        是否使用 GPU 加速
        gpu_id:         GPU 设备 ID
        num_threads:    CPU 搜索线程数，0 表示使用系统默认
    """

    # 默认配置字典
    DEFAULT_CONFIG: Dict[str, Any] = {
        'dimension': 128,  # 向量维度（必须与数据一致）
        'index_type': 'flat',  # 索引类型：flat/ivf/hnsw/pq/ivf_pq
        'metric': 'l2',  # 距离度量：l2/ip/cosine
        'index_dir': './faiss',  # 默认索引文件存储目录
        'index_name': 'faiss.index',  # 索引文件名
        'auto_persist': False,  # 是否自动持久化
        'nlist': 100,  # IVF 聚类中心数量
        'nprobe': 10,  # IVF 搜索探测数量
        'm': 16,  # PQ 子向量数量
        'nbits': 8,  # PQ 比特数
        'M': 32,  # HNSW 节点连接数
        'efSearch': 64,  # HNSW 搜索参数
        'use_gpu': False,  # 是否使用 GPU
        'gpu_id': 0,  # GPU 设备 ID
        'num_threads': 0,  # CPU 线程数
    }

    # 支持的索引类型列表
    INDEX_TYPES = ['flat', 'ivf', 'hnsw', 'pq', 'ivf_pq']
    # 支持的距离度量类型列表
    METRICS = ['l2', 'ip', 'cosine']

    def __init__(self, config=None):
        """
        初始化 FAISS 管理器（懒加载模式）

        Args:
            config: 配置字典，可选参数，用于覆盖默认配置
                    未提供的配置项将使用 DEFAULT_CONFIG 中的默认值
        """
        self._index = None  # FAISS 索引实例
        self._dimension = None  # 向量维度缓存
        self._gpu_resources = None  # GPU 资源对象
        self._index_info: Dict[int, Any] = {}  # 索引信息字典，存储向量的元数据 {id: metadata}
        self._lock = threading.RLock()  # 线程锁，保证线程安全
        super().__init__(config)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        获取默认配置字典的副本

        Returns:
            Dict[str, Any]: 默认配置字典的深拷贝
        """
        return FAISSManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def config_help() -> str:
        """
        获取配置帮助信息，详细说明每个配置项的含义和用法

        Returns:
            str: 格式化的配置帮助文档
        """
        logger.info("_help_doc")
        return _help_doc

    def _create_index(self) -> Any:
        """
        创建 FAISS 索引实例

        根据配置中的 index_type 和 metric 参数创建对应的索引类型，
        并应用相关的索引参数（nlist, nprobe, M, efSearch 等）。

        Returns:
            faiss.Index: FAISS 索引实例

        Raises:
            ValueError: 当 index_type 不在支持列表中时
        """
        # 获取配置参数
        dimension = self.get_config('dimension', 128)  # 向量维度
        index_type = self.get_config('index_type', 'flat')  # 索引类型
        metric = self.get_config('metric', 'l2')  # 距离度量方式
        is_inner_product = metric in ('ip', 'cosine')  # 是否使用内积度量

        index = None  # 索引实例
        # 根据索引类型创建对应的索引
        if index_type == 'flat':
            # Flat 索引：暴力搜索，精度最高
            if is_inner_product:
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlatL2(dimension)

        elif index_type == 'ivf':
            # IVF 索引：倒排索引，需要训练
            num_clusters = self.get_config('nlist', 100)  # 聚类中心数量
            if is_inner_product:
                quantizer = faiss.IndexFlatIP(dimension)  # 量化器
                index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters, faiss.METRIC_L2)

        elif index_type == 'hnsw':
            # HNSW 索引：分层可导航小世界图
            max_connections = self.get_config('M', 32)  # 每个节点的最大连接数
            if is_inner_product:
                index = faiss.IndexHNSWFlat(dimension, max_connections, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexHNSWFlat(dimension, max_connections, faiss.METRIC_L2)

        elif index_type == 'pq':
            # PQ 索引：产品量化
            num_subvectors = self.get_config('m', 16)  # 子向量数量
            bits_per_subvector = self.get_config('nbits', 8)  # 每个子向量的比特数
            index = faiss.IndexPQ(
                dimension,
                num_subvectors,
                bits_per_subvector,
                faiss.METRIC_INNER_PRODUCT if is_inner_product else faiss.METRIC_L2
            )

        elif index_type == 'ivf_pq':
            # IVF_PQ 索引：倒排索引 + 产品量化
            num_clusters = self.get_config('nlist', 100)
            num_subvectors = self.get_config('m', 16)
            bits_per_subvector = self.get_config('nbits', 8)
            if is_inner_product:
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFPQ(
                    quantizer, dimension, num_clusters, num_subvectors, bits_per_subvector,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFPQ(
                    quantizer, dimension, num_clusters, num_subvectors, bits_per_subvector,
                    faiss.METRIC_L2
                )

        else:
            raise ValueError(f"不支持的索引类型: {index_type}，支持的类型: {self.INDEX_TYPES}")

        # 始终使用 IndexIDMap 包装索引，支持自定义 ID
        index = faiss.IndexIDMap(index)

        # 设置 IVF 相关参数
        num_probes = self.get_config('nprobe', 10)
        if hasattr(index, 'nprobe'):
            index.nprobe = num_probes

        # 设置 HNSW 相关参数
        ef_search = self.get_config('efSearch', 64)
        if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
            index.hnsw.efSearch = ef_search

        # 设置多线程
        num_threads = self.get_config('num_threads', 0)
        if num_threads > 0:
            faiss.omp_set_num_threads(num_threads)

        logger.info(f"创建 FAISS 索引: 类型={index_type}, 维度={dimension}, 度量={metric}")
        return index

    def _connect(self) -> bool:
        """
        初始化 FAISS 索引（懒加载核心方法）

        优先尝试从配置的路径加载已有的索引文件，
        如果加载失败或文件不存在，则创建新索引。

        Returns:
            bool: 初始化是否成功
        """
        index_directory = self.get_config('index_dir', './faiss')  # 索引存储目录
        index_filename = self.get_config('index_name', 'faiss.index')  # 索引文件名

        # 构建索引文件路径
        index_path = str(os.path.join(index_directory, index_filename))

        if index_path and os.path.exists(index_path):
            # 从本地文件加载
            try:
                self._index = faiss.read_index(index_path)
                # 确保索引被 IndexIDMap 包装
                if not isinstance(self._index, faiss.IndexIDMap):
                    # 使用 IndexIDMap2 包装已有的非空索引
                    if self._index.ntotal > 0:
                        self._index = faiss.IndexIDMap2(self._index)
                    else:
                        self._index = faiss.IndexIDMap(self._index)
                self._dimension = self._index.d
                self._is_connected = True
                logger.info(f"从本地文件加载 FAISS 索引: {index_path}")

                # 尝试加载索引信息字典
                index_info_path = f"{index_path}.info"
                if os.path.exists(index_info_path):
                    with open(index_info_path, 'rb') as f:
                        self._index_info = pickle.load(f)
                    logger.info(f"加载索引信息字典，共 {len(self._index_info)} 条")

                return True
            except Exception as e:
                logger.warning(f"加载本地索引失败: {e}，创建新索引")
                self._index = self._create_index()

        else:
            # 创建新索引
            self._index = self._create_index()

        # 设置维度并标记已连接
        self._dimension = self.get_config('dimension', 128)
        self._is_connected = True
        logger.info("FAISS 索引初始化成功")
        return True

    def _reconnect_if_needed(self) -> bool:
        """
        检查并重新连接（如果需要）

        FAISS 索引是本地内存结构，不需要网络重连，
        此方法仅在索引未初始化时进行初始化。

        Returns:
            bool: 是否已连接
        """
        if self._index is None:
            return self._connect()
        return True

    def _normalize_cosine(self, vectors: np.ndarray) -> np.ndarray:
        """
        对向量进行余弦归一化

        当 metric 为 'cosine' 时，需要对向量进行 L2 归一化，
        这样内积运算的结果就等于余弦相似度。

        Args:
            vectors: 待归一化的向量数组

        Returns:
            np.ndarray: 归一化后的向量数组
        """
        if self.get_config('metric') == 'cosine':
            faiss.normalize_L2(vectors)
        return vectors

    @property
    def index(self):
        """
        获取 FAISS 索引实例（懒加载）

        如果索引尚未初始化，会自动调用 _ensure_connected() 进行初始化。

        Returns:
            faiss.Index: FAISS 索引实例
        """
        self._ensure_connected()
        return self._index

    @property
    def dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        if self._dimension is None:
            self._dimension = self.get_config('dimension', 128)
        return self._dimension

    @property
    def vec_total(self) -> int:
        """
        获取索引中的向量总数

        Returns:
            int: 向量数量
        """
        self._ensure_connected()
        return self._index.ntotal

    @property
    def index_info(self):
        return self._index_info

    def add(self, vectors: np.ndarray):
        """
        添加向量到索引

        Args:
            vectors: 向量数组，形状必须为 (n, dimension)
        Raises:
            ValueError: 向量维度不匹配、ids 长度不匹配等
        """
        with self._lock:
            self._ensure_connected()

            if vectors.ndim != 2:
                raise ValueError(f"向量必须是二维数组，当前形状: {vectors.shape}")

            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"向量维度不匹配，期望: {self.dimension}, 实际: {vectors.shape[1]}"
                )

            vectors = vectors.astype('float32')
            vectors = self._normalize_cosine(vectors)

            if not self._index.is_trained:
                logger.info("索引未训练，先进行训练")
                self._index.train(vectors)

            # 自增id, 自动生成唯一 ID（从当前最大 ID + 1 开始）
            start_id = self._index.ntotal
            ids = np.arange(start_id, start_id + len(vectors)).astype(np.int64)

            self._index.add_with_ids(vectors, ids)

            logger.info(f"添加 {len(vectors)} 个向量到索引")

            if self.get_config('auto_persist', False):
                self.save()

    def add_with_metadata(self, vectors: np.ndarray, metadata: List[Any]):
        """
        添加带元数据的向量到索引，同时构建索引信息字典

        Args:
            vectors: 向量数组，形状必须为 (n, dimension)
            metadata: 元数据列表，长度必须为 n

        Raises:
            ValueError: 向量维度不匹配、ids/metadata 长度不匹配等
        """
        with self._lock:
            self._ensure_connected()

            if vectors.ndim != 2:
                raise ValueError(f"向量必须是二维数组，当前形状: {vectors.shape}")

            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"向量维度不匹配，期望: {self.dimension}, 实际: {vectors.shape[1]}"
                )

            if len(metadata) != len(vectors):
                raise ValueError("vectors 和 metadata 长度不匹配")

            vectors = vectors.astype('float32')
            vectors = self._normalize_cosine(vectors)

            if not self._index.is_trained:
                logger.info("索引未训练，先进行训练")
                self._index.train(vectors)

            # 自增id, 自动生成唯一 ID（从当前最大 ID + 1 开始）
            start_id = self._index.ntotal
            ids = np.arange(start_id, start_id + len(vectors)).astype(np.int64)
            self._index.add_with_ids(vectors, ids)

            for idx, meta in zip(ids, metadata):
                self._index_info[int(idx)] = meta

            logger.info(f"添加 {len(vectors)} 个带元数据的向量到索引")

            if self.get_config('auto_persist', False):
                self.save()

    def get_metadata(self, indices: np.ndarray) -> List[Any]:
        """
        根据索引数组获取对应的元数据列表

        Args:
            indices: 索引数组，可以是一维或二维数组

        Returns:
            List[Any]: 元数据列表，与输入索引一一对应
        """
        result = []
        flat_indices = indices.flatten()
        for idx in flat_indices:
            result.append(self._index_info.get(int(idx), ""))
        return result

    def search(self, query: np.ndarray, k: int = 5) -> tuple:
        """
        搜索最近邻向量

        Args:
            query: 查询向量，可以是一维数组 (dimension,) 或二维数组 (n, dimension)
            k: 返回的最近邻数量，默认值为 5

        Returns:
            tuple: (distances, indices)，分别为距离数组和索引数组

        Raises:
            ValueError: 查询向量维度不匹配
        """
        self._ensure_connected()

        # 处理一维查询向量
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 验证查询向量维度
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"查询向量维度不匹配，期望: {self.dimension}, 实际: {query.shape[1]}"
            )

        # 转换为 float32 类型并进行余弦归一化
        query = query.astype('float32')
        query = self._normalize_cosine(query)

        # 执行搜索
        distances, indices = self._index.search(query, k)

        logger.debug(f"搜索完成，返回 {k} 个最近邻")
        return distances, indices

    def search_with_info(self, query: np.ndarray, k: int = 5) -> tuple:
        """
        搜索最近邻向量并返回对应的文本

        Args:
            query: 查询向量，可以是一维数组 (dimension,) 或二维数组 (n, dimension)
            k: 返回的最近邻数量，默认值为 5

        Returns:
            tuple: (distances, indices, texts)，分别为距离数组、索引数组和文本列表

        Raises:
            ValueError: 查询向量维度不匹配
        """
        distances, indices = self.search(query, k)
        texts = self.get_metadata(indices)
        return distances, indices, texts

    def remove(self, ids: np.ndarray):
        """
        从索引中移除向量，同时删除对应的索引信息

        Args:
            ids: 要移除的向量 ID 数组，形状必须为 (n,)

        Raises:
            ValueError: ID 类型不正确
        """
        with self._lock:
            self._ensure_connected()

            if ids.dtype != np.int64:
                ids = ids.astype(np.int64)

            self._index.remove_ids(ids)

            for idx in ids:
                self._index_info.pop(int(idx), None)

            logger.info(f"移除 {len(ids)} 个向量")

            if self.get_config('auto_persist', False):
                self.save()

    def train(self, vectors: np.ndarray):
        """
        训练索引（适用于需要训练的索引类型，如 IVF、PQ、IVF_PQ）

        Args:
            vectors: 训练向量数组，形状必须为 (n, dimension)

        Raises:
            ValueError: 向量维度不匹配
        """
        self._ensure_connected()

        # 验证向量维度
        if vectors.ndim != 2:
            raise ValueError(f"向量必须是二维数组，当前形状: {vectors.shape}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配，期望: {self.dimension}, 实际: {vectors.shape[1]}"
            )

        # 转换为 float32 类型并进行余弦归一化
        vectors = vectors.astype('float32')
        vectors = self._normalize_cosine(vectors)

        # 如果索引未训练，进行训练
        if not self._index.is_trained:
            logger.info("开始训练索引...")
            self._index.train(vectors)
            logger.info("索引训练完成")

    def save(self, path: Optional[str] = None):
        """
        保存索引到文件（包含文本索引字典）

        Args:
            path: 保存路径，可以是相对路径或绝对路径。
                  如果为 None，则保存到配置中指定的默认路径（index_dir/index_name）
        """
        with self._lock:
            self._ensure_connected()

            # 如果未指定路径，使用配置中的默认路径
            if path is None:
                index_directory = self.get_config('index_dir', './faiss')
                index_filename = self.get_config('index_name', 'faiss.index')
                path = os.path.join(index_directory, index_filename)

            # 确保目录存在
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # 写入索引文件
            faiss.write_index(self._index, path)
            logger.info(f"索引已保存到: {path}")

            # 保存索引信息字典
            if self._index_info:
                index_info_path = f"{path}.info"
                with open(index_info_path, 'wb') as f:
                    pickle.dump(self._index_info, f)
                logger.info(f"索引信息字典已保存到: {index_info_path}")

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None):
        """
        从文件加载索引（包含文本索引字典）

        Args:
            path: 索引文件路径，本地路径
            config: 配置字典，可选参数

        Returns:
            FAISSManager: 加载了索引的管理器实例
        """
        if config is None:
            config = {}

        # 解析目录和文件名
        config['index_dir'] = os.path.dirname(path) or './faiss'
        config['index_name'] = os.path.basename(path)

        # 创建实例并确保连接
        manager = cls(config)
        manager._ensure_connected()
        return manager

    def reconstruct(self, index: int) -> np.ndarray:
        """
        重建指定索引位置的向量

        Args:
            index: 向量在索引中的位置（从 0 开始）

        Returns:
            np.ndarray: 重建的向量，形状为 (dimension,)

        Raises:
            ValueError: 当前索引类型不支持重建
        """
        self._ensure_connected()

        # 检查是否支持重建
        if not hasattr(self._index, 'reconstruct'):
            raise ValueError(
                f"当前索引类型 {self.get_config('index_type')} 不支持重建"
            )

        return self._index.reconstruct(index)

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        """
        批量重建向量

        Args:
            start: 起始索引位置（从 0 开始）
            count: 重建的向量数量

        Returns:
            np.ndarray: 重建的向量数组，形状为 (count, dimension)

        Raises:
            ValueError: 当前索引类型不支持批量重建
        """
        self._ensure_connected()

        # 检查是否支持批量重建
        if not hasattr(self._index, 'reconstruct_n'):
            raise ValueError(
                f"当前索引类型 {self.get_config('index_type')} 不支持批量重建"
            )

        return self._index.reconstruct_n(start, count)

    def range_search(self, query: np.ndarray, radius: float = 10.0) -> tuple:
        """
        范围搜索：查找距离查询向量在指定半径内的所有向量

        Args:
            query: 查询向量，可以是一维数组 (dimension,) 或二维数组 (n, dimension)
            radius: 搜索半径，默认值为 10.0

        Returns:
            tuple: (lims, distances, indices)
                   lims: 每个查询的结果数量边界
                   distances: 距离数组
                   indices: 索引数组

        Raises:
            ValueError: 查询向量维度不匹配或当前索引类型不支持范围搜索
        """
        self._ensure_connected()

        # 处理一维查询向量
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 验证查询向量维度
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"查询向量维度不匹配，期望: {self.dimension}, 实际: {query.shape[1]}"
            )

        # 转换为 float32 类型并进行余弦归一化
        query = query.astype('float32')
        query = self._normalize_cosine(query)

        # 检查是否支持范围搜索
        if not hasattr(self._index, 'range_search'):
            raise ValueError(
                f"当前索引类型 {self.get_config('index_type')} 不支持范围搜索"
            )

        return self._index.range_search(query, radius)

    def merge_from(self, other: 'FAISSManager', shift_ids: bool = False):
        """
        合并另一个索引到当前索引

        注意：faiss.merge_into 只支持 IVF 类型索引（ivf, ivf_pq），
        对于其他类型的索引，采用手动合并方式（重建并添加向量）。

        Args:
            other: 另一个 FAISSManager 实例
            shift_ids: 是否偏移 ID，仅对 IVF 类型索引有效

        Raises:
            ValueError: 另一个索引未初始化或维度不匹配
        """
        self._ensure_connected()

        # 验证另一个索引
        if other._index is None:
            raise ValueError("另一个索引未初始化")

        # 验证维度匹配
        if self._index.d != other._index.d:
            raise ValueError(
                f"索引维度不匹配，当前: {self._index.d}, 另一个: {other._index.d}"
            )

        index_type = self.get_config('index_type', 'flat')

        # 检查是否为 IVF 类型索引
        if index_type not in ['ivf', 'ivf_pq']:
            logger.warning(
                f"索引合并只支持 IVF 类型索引，当前类型: {index_type}"
            )
            logger.info("采用手动合并方式：重建并添加另一个索引的向量")

            try:
                total_count = other._index.ntotal
                if total_count > 0 and hasattr(other._index, 'reconstruct_n'):
                    # 重建另一个索引的所有向量并添加到当前索引
                    vectors = other.reconstruct_n(0, total_count)
                    self.add(vectors)
                    logger.info(f"手动合并完成，当前向量数量: {self._index.ntotal}")
                else:
                    raise ValueError("无法重建另一个索引的向量")
            except Exception as e:
                raise ValueError(f"手动合并失败: {e}")
            return

        # IVF 类型索引使用 faiss.merge_into
        faiss.merge_into(self._index, other._index, shift_ids)
        logger.info(f"合并完成，当前向量数量: {self._index.ntotal}")

    def to_gpu(self, gpu_id: int = 0):
        """
        将索引转移到 GPU

        Args:
            gpu_id: GPU 设备 ID，默认值为 0

        Raises:
            Exception: 转移失败时
        """
        self._ensure_connected()

        try:
            # 创建 GPU 资源
            gpu_resources = faiss.StandardGpuResources()
            # 将 CPU 索引转换为 GPU 索引
            self._index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, self._index)
            self._gpu_resources = gpu_resources
            logger.info(f"索引已转移到 GPU: {gpu_id}")
        except Exception as e:
            logger.error(f"转移到 GPU 失败: {e}")
            raise

    def to_cpu(self):
        """将索引从 GPU 转移回 CPU"""
        self._ensure_connected()

        # 检查是否为 GPU 索引
        if not faiss.is_gpu_index(self._index):
            return

        # 将 GPU 索引转换为 CPU 索引
        self._index = faiss.index_gpu_to_cpu(self._index)
        self._gpu_resources = None
        logger.info("索引已转移回 CPU")

    def set_num_threads(self, num_threads: int):
        """
        设置 CPU 搜索线程数

        Args:
            num_threads: 线程数，0 表示使用系统默认
        """
        faiss.omp_set_num_threads(num_threads)
        logger.info(f"设置线程数: {num_threads}")

    def get_num_threads(self) -> int:
        """
        获取当前 CPU 搜索线程数

        Returns:
            int: 当前线程数
        """
        return faiss.omp_get_max_threads()

    def close(self):
        """关闭连接，清理索引引用和 GPU 资源"""
        self._index = None
        self._gpu_resources = None
        self._index_info = {}
        self._is_connected = False
        logger.info("FAISS 索引已关闭")

    def is_connected(self) -> bool:
        """
        检查索引是否已初始化

        Returns:
            bool: 是否已连接（初始化）
        """
        return self._index is not None and self._is_connected

    def __repr__(self) -> str:
        """
        返回对象的字符串表示

        Returns:
            str: 包含维度、类型、向量数量和状态的字符串
        """
        status = 'connected' if self.is_connected() else 'disconnected'
        dimension = self._dimension or self.get_config('dimension', '?')
        index_type = self.get_config('index_type', 'flat')
        total_count = self._index.ntotal if self._index else '?'
        meta_count = len(self._index_info)
        return (
            f"FAISSManager(dimension={dimension}, type={index_type}, "
            f"ntotal={total_count}, metadata={meta_count}, status={status})"
        )
