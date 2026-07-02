# -*- coding: utf-8 -*-
# @Date  : 2026/6/22 17:10
# @Author: Finks
# @Desc  : 文本处理工具模块

import re
from typing import List, Optional, Tuple, Any


def remove_extra_spaces(text: str) -> str:
    """
    移除多余的空格（多个连续空格合并为一个）
    
    Args:
        text: 输入字符串
    
    Returns:
        str: 处理后的字符串
    """
    return re.sub(r'\s+', ' ', text).strip()


def to_snake_case(text: str) -> str:
    """
    将字符串转换为蛇形命名（snake_case）
    
    Args:
        text: 输入字符串
    
    Returns:
        str: 蛇形命名字符串
    """
    # 处理驼峰命名
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    # 处理空格和连字符
    text = re.sub(r'[\s\-]+', '_', text)
    # 转小写并移除连续下划线
    return re.sub(r'_+', '_', text).lower()


def to_camel_case(text: str, capitalize_first: bool = False) -> str:
    """
    将字符串转换为驼峰命名
    
    Args:
        text: 输入字符串
        capitalize_first: 是否大写首字母（大驼峰），默认为 False（小驼峰）
    
    Returns:
        str: 驼峰命名字符串
    """
    # 分割字符串
    words = re.split(r'[\s_\-]+', text)
    # 首字母大写
    words = [word.capitalize() for word in words if word]

    if not words:
        return ''

    if capitalize_first:
        return ''.join(words)
    else:
        return words[0].lower() + ''.join(words[1:])


def extract_numbers(text: str) -> List[float]:
    """
    从字符串中提取所有数字
    
    Args:
        text: 输入字符串
    
    Returns:
        List[float]: 提取的数字列表
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]



def is_valid_email(email: str) -> bool:
    """
    检查是否为有效的邮箱地址
    
    Args:
        email: 邮箱字符串
    
    Returns:
        bool: 是否有效
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_url(url: str) -> bool:
    """
    检查是否为有效的 URL
    
    Args:
        url: URL 字符串
    
    Returns:
        bool: 是否有效
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return re.match(pattern, url) is not None


def split_text(text: str, delimiter: str = '\n', max_parts: Optional[int] = None) -> List[str]:
    """
    分割文本
    
    Args:
        text: 输入文本
        delimiter: 分隔符，默认为换行符
        max_parts: 最大分割数量，默认为 None（不限制）
    
    Returns:
        List[str]: 分割后的文本列表
    """
    parts = text.split(delimiter)

    if max_parts is not None and len(parts) > max_parts:
        parts = parts[:max_parts]
        # 将剩余部分合并到最后一个元素
        if max_parts > 0:
            remaining = delimiter.join(text.split(delimiter)[max_parts - 1:])
            parts[-1] = remaining

    return parts


def clean_text(text: str) -> str:
    """
    清洗文本，移除特殊字符
    
    Args:
        text: 输入文本
    
    Returns:
        str: 清洗后的文本
    """
    # 移除不可打印字符
    text = ''.join(char for char in text if char.isprintable())
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    return text.strip()


def format_number(number: float, decimals: int = 2) -> str:
    """
    格式化数字，保留指定位数的小数
    
    Args:
        number: 输入数字
        decimals: 小数位数，默认为 2
    
    Returns:
        str: 格式化后的字符串
    """
    return f"{number:.{decimals}f}"


def mask_sensitive(text: str, pattern: str = r'(\d{3})\d{4}(\d{4})', replacement: str = r'\1****\2') -> str:
    """
    敏感信息脱敏处理
    
    Args:
        text: 输入文本
        pattern: 匹配模式，默认为手机号模式
        replacement: 替换模板
    
    Returns:
        str: 脱敏后的文本
    """
    return re.sub(pattern, replacement, text)


if __name__ == '__main__':
    pass
    result = to_snake_case("HelloWorld")
    print(result)
    print(mask_sensitive('1234567890'))
