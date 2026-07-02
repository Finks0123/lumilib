# -*- coding: utf-8 -*-
# @Date  : 2026/7/1 18:40
# @Author: Finks
# @Desc  : 
"""

"""
import numpy as np

"""
计算向量之间的余弦相似度
"""


def cos_similarity(vec: np.ndarray, vec_set: np.ndarray) -> np.ndarray:
    """
    :param vec:     单个向量[d,]或多个向量[m, d]
    :param vec_set: 向量集合[n, d]
    :return:
        - 若 vec为[d,]返回[n,]的相似度数组
        - 若 vec为[m, d]返回[m, n]的相似度矩阵
    """
    import numpy as np
    # 转np数组
    vec = np.asarray(vec, dtype=np.float32)
    vec_set = np.asarray(vec_set, dtype=np.float32)
    if vec_set.ndim != 2:
        raise ValueError(f"vec_set must be 2D array with shape (n, d), got {vec_set.shape}")

    n, d = vec_set.shape

    #  避免除以0
    def _safe_norm(x, axis=None, keepdims=False):
        eps = 1e-8
        norms = np.linalg.norm(x, axis=axis, keepdims=keepdims)
        norms = np.maximum(norms, eps)
        return norms

    if vec.ndim == 1:
        if vec.shape[0] != d:
            raise ValueError(f"dimension mismatch: vec has dim {vec.shape[0]}, vec_set has dim {d}")

        # dot products (n,)
        dots = vec_set.dot(vec)  # (n,)

        # norms
        vec_norm = _safe_norm(vec)
        set_norms = _safe_norm(vec_set, axis=1)

        sims = dots / (set_norms * vec_norm)
        return sims.astype(np.float32)
    elif vec.ndim == 2:
        m, dv = vec.shape
        if dv != d:
            raise ValueError(f"dimension mismatch: vec has dim {dv}, vec_set has dim {d}")
        # 为了控制内存，当 m 或 n 很大时分块计算
        CHUNK_SIZE = 4096  # 根据内存与维度可调（4096 在多数场景下是合理的起点）
        if m * n <= 10_000_000:  # 若结果矩阵较小，直接计算（避免循环开销）
            dots = vec.dot(vec_set.T)  # (m, n)
            vec_norms = _safe_norm(vec, axis=1)[:, None]  # (m,1)
            set_norms = _safe_norm(vec_set, axis=1)[None, :]  # (1,n)
            sims = dots / (vec_norms * set_norms)
            return sims.astype(np.float32)
        else:
            # 分块计算以节省峰值内存
            out = np.empty((m, n), dtype=np.float32)
            set_norms = _safe_norm(vec_set, axis=1)  # (n,)
            for start in range(0, m, CHUNK_SIZE):
                end = min(m, start + CHUNK_SIZE)
                chunk = vec[start:end]  # (chunk_size, d)
                dots_chunk = chunk.dot(vec_set.T)  # (chunk_size, n)
                chunk_norms = _safe_norm(chunk, axis=1)[:, None]  # (chunk_size,1)
                sims_chunk = dots_chunk / (chunk_norms * set_norms[None, :])
                out[start:end] = sims_chunk.astype(np.float32)
            return out

    else:
        raise ValueError("vec must be 1D or 2D numpy array")

