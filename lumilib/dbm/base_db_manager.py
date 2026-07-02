# -*- coding: utf-8 -*-
# @Date  : 2026/6/16 18:30
# @Author: Finks
# @Desc  : 数据库管理器基类

from typing import Any, Dict, Optional

from pydantic import BaseModel

from lumilib.common.logger import logger


class BaseDBManager(object):
    """
    数据库管理器基类（懒加载模式）

    特性：
        - 懒加载连接：只在首次访问数据库时才建立连接
        - 配置管理：支持默认配置和深度定制，支持 Pydantic 配置模型
        - 自动重连：连接断开时自动尝试重连
        - 健康检查：定期检查连接健康状态

    使用方式：
        # 使用配置字典
        db = MongoDBManager({'mongo_url': '...', 'db_name': '...'})

        # 使用 Pydantic 配置模型（推荐）
        from lumilib.dbm import MongoDBConfig
        config = MongoDBConfig(mongo_url='...', db_name='...')
        db = MongoDBManager(config)
    """

    # 子类必须定义默认配置
    DEFAULT_CONFIG: Dict[str, Any] = {}

    def __init__(self, config=None):
        self._config: Dict[str, Any] = {}
        self._is_connected: bool = False

        if self.DEFAULT_CONFIG:
            self._config.update(self.DEFAULT_CONFIG)
        if config:
            if isinstance(config, BaseModel):
                self._config.update(config.model_dump())
            elif isinstance(config, dict):
                self._config.update(config)
            else:
                raise ValueError("配置必须是字典或 Pydantic 模型类型")

    def _reconnect_if_needed(self) -> bool:
        """
        如果连接断开，尝试重新连接（子类必须实现）
        """
        raise NotImplementedError("子类必须实现 _reconnect_if_needed() 方法")

    def close(self):
        """
        关闭连接（子类应该重写）
        """
        self._is_connected = False
        logger.info(f"{self.__class__.__name__} 连接已关闭")

    def _ensure_connected(self) -> bool:
        """
        确保已连接（懒加载核心方法）

        Returns:
            bool: 是否已连接
        """
        if not self._is_connected:
            return self._connect()
        return self._reconnect_if_needed()

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置字典的只读视图"""
        return dict(self._config)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        Args:
            key: 配置键（支持点号分隔的嵌套键，如 'db.name'）
            default: 默认值
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set_config(self, config):
        """设置配置（仅允许未连接时调用）"""
        if self._is_connected:
            raise RuntimeError("已连接数据库，禁止修改配置。请先调用 close()")
        if isinstance(config, BaseModel):
            self._config.update(config.model_dump())
        elif isinstance(config, dict):
            self._config.update(config)
        else:
            raise ValueError("配置必须是字典或 Pydantic 模型类型")
        logger.debug('set config success!')

    def update_config(self, config):
        """更新配置（仅允许未连接时调用）"""
        self.set_config(config)

    @staticmethod
    def config_help() -> str:
        """
        获取配置帮助信息（子类必须实现）

        Returns:
            str: 帮助文档字符串
        """
        raise NotImplementedError('子类需实现 config_help()方法')

    def _connect(self) -> bool:
        """
        建立连接（子类必须实现）
        """
        raise NotImplementedError("子类必须实现 _connect() 方法")