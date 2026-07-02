# -*- coding: utf-8 -*-
# @Date  : 2026/7/1 18:38
# @Author: Finks
# @Desc  : 
"""

"""
# -*- coding: utf-8 -*-
# @Date  : 2026/5/29 13:33
# @Author: Finks
# @Desc  :
"""
文本编码器

"""
import numpy as np
from lumilib.common.logger import logger


class TextEncoder(object):
    def __init__(self, path):
        self.path = path
        # 加载预训练模型
        from sentence_transformers import SentenceTransformer
        logger.info(f"Embedding模型 load from:  [{path}]")
        self.model = SentenceTransformer(str(self.path))

    def encode(self, text: str | list[str]) -> np.ndarray:
        """embedding"""
        return self.model.encode(text, normalize_embeddings=True)
