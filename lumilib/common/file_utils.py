# -*- coding: utf-8 -*-
# @Date  : 2026/6/22 17:05
# @Author: Finks
# @Desc  : 文件操作工具模块

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Union



def write_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    写入文件内容（覆盖写入）
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
        encoding: 编码格式，默认为 utf-8
    
    Raises:
        IOError: 文件写入失败
    """
    file_path = str(file_path)
    ensure_dir(os.path.dirname(file_path))
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"写入文件失败: {e}")


def append_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    追加内容到文件
    
    Args:
        file_path: 文件路径
        content: 要追加的内容
        encoding: 编码格式，默认为 utf-8
    
    Raises:
        IOError: 文件追加失败
    """
    file_path = str(file_path)
    ensure_dir(os.path.dirname(file_path))
    try:
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"追加文件失败: {e}")


def list_files(directory: Union[str, Path], pattern: str = '*', recursive: bool = False) -> List[str]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式，默认为 '*'
        recursive: 是否递归搜索，默认为 False
    
    Returns:
        List[str]: 文件路径列表
    """
    directory = str(directory)
    if recursive:
        files = [str(p) for p in Path(directory).rglob(pattern)]
    else:
        files = [str(p) for p in Path(directory).glob(pattern)]
    return sorted(files)


def ensure_dir(directory: Union[str, Path]) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    directory = str(directory)
    if directory:
        os.makedirs(directory, exist_ok=True)


def remove_file(file_path: Union[str, Path]) -> None:
    """
    删除文件
    
    Args:
        file_path: 文件路径
    
    Raises:
        FileNotFoundError: 文件不存在
        IOError: 删除失败
    """
    file_path = str(file_path)
    try:
        os.remove(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise IOError(f"删除文件失败: {e}")


def copy_file(src_path: Union[str, Path], dst_path: Union[str, Path], overwrite: bool = True) -> None:
    """
    复制文件
    
    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        overwrite: 是否覆盖已存在的文件，默认为 True
    
    Raises:
        FileNotFoundError: 源文件不存在
        IOError: 复制失败
    """
    src_path = str(src_path)
    dst_path = str(dst_path)
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源文件不存在: {src_path}")
    
    if os.path.exists(dst_path) and not overwrite:
        raise IOError(f"目标文件已存在: {dst_path}")
    
    ensure_dir(os.path.dirname(dst_path))
    
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        raise IOError(f"复制文件失败: {e}")


def move_file(src_path: Union[str, Path], dst_path: Union[str, Path], overwrite: bool = True) -> None:
    """
    移动文件
    
    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        overwrite: 是否覆盖已存在的文件，默认为 True
    
    Raises:
        FileNotFoundError: 源文件不存在
        IOError: 移动失败
    """
    src_path = str(src_path)
    dst_path = str(dst_path)
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源文件不存在: {src_path}")
    
    if os.path.exists(dst_path) and not overwrite:
        raise IOError(f"目标文件已存在: {dst_path}")
    
    ensure_dir(os.path.dirname(dst_path))
    
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        raise IOError(f"移动文件失败: {e}")


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
    
    Returns:
        bool: 文件是否存在
    """
    return os.path.exists(str(file_path))


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        int: 文件大小（字节）
    
    Raises:
        FileNotFoundError: 文件不存在
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return os.path.getsize(file_path)


def get_file_hash(file_path: Union[str, Path], hash_algorithm: str = 'md5') -> str:
    """
    计算文件的哈希值
    
    Args:
        file_path: 文件路径
        hash_algorithm: 哈希算法，支持 md5、sha1、sha256，默认为 md5
    
    Returns:
        str: 文件的哈希值
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的哈希算法
    """
    file_path = str(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256
    }
    
    if hash_algorithm not in algorithms:
        raise ValueError(f"不支持的哈希算法: {hash_algorithm}")
    
    hash_obj = algorithms[hash_algorithm]()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        raise IOError(f"计算文件哈希失败: {e}")


def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> dict:
    """
    读取 JSON 文件
    
    Args:
        file_path: 文件路径
        encoding: 编码格式
    
    Returns:
        dict: JSON 数据
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 解析失败
    """
    import json
    content = read_file(file_path, encoding)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}")


def write_json(file_path: Union[str, Path], data: dict, encoding: str = 'utf-8', indent: int = 4) -> None:
    """
    写入 JSON 文件
    
    Args:
        file_path: 文件路径
        data: 要写入的数据
        encoding: 编码格式
        indent: 缩进空格数
    
    Raises:
        IOError: 文件写入失败
    """
    import json
    content = json.dumps(data, ensure_ascii=False, indent=indent)
    write_file(file_path, content, encoding)

