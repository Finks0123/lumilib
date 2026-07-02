# -*- coding: utf-8 -*-
# @Date  : 2026/6/22 16:56
# @Author: Finks
# @Desc  : 日志工具模块
"""
一切为了简约，易懂，优雅。

"""

import sys
import os
from loguru import logger


def _format_record(record, color=True):
    """
    格式化日志记录
    
    Args:
        record: 日志记录对象
        color: 是否使用颜色
    
    Returns:
        str: 格式化后的日志字符串
    """
    name = record.get("name", "")
    function = record.get("function", "")
    line = record.get("line", "")
    location = f"{name}.{function}:{line}"

    # padding到最固定长度, 维持日志美观
    max_len = 40
    if len(location) > max_len:
        location = location[:max_len - 2] + ".."
    else:
        location = location + " " * (max_len - len(location))
    record["extra"]["location"] = location

    if color:
        _format_pat = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8.8}</level> | "
                       "<cyan>{extra[location]}</cyan> | "
                       "<level>{message}\n{exception}</level>")
    else:
        _format_pat = ("{time:YYYY-MM-DD HH:mm:ss} | "
                       "{level: <8.8} | "
                       "{extra[location]} | "
                       "{message}\n{exception}")
    return _format_pat


def console_format(record):
    """终端输出格式（带颜色）"""
    return _format_record(record, color=True)


def file_format(record):
    """文件输出格式（无颜色）"""
    return _format_record(record, color=False)


# 移除默认的 handler
# 终端输出（带颜色）
logger.remove()
logger.add(
    sys.stderr,
    format=console_format,
    colorize=True,
    level="DEBUG"
)


def set_log_dir(log_dir: str):
    """
    设定日志目录
    :param log_dir:
    :return:
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        raise f"文件目录错误:{log_dir}"
    # 如果设定了目录
    logger.add(
        os.path.join(log_dir, "app.log"),
        format=file_format,
        rotation="00:00",
        retention="3 days",
        compression="zip",
        level="INFO",
        encoding="utf-8",
    )


if __name__ == '__main__':
    def test_long_name_for_cut_split_function_name():
        logger.info("测试超长路径截断效果")
        logger.debug("这是一条 DEBUG（不会记录到文件）")
        logger.error("示例错误日志", Exception("示例异常"))
        logger.warning("警告⚠️")


    logger.info('普通日志')
    test_long_name_for_cut_split_function_name()
