# -*- coding: utf-8 -*-
# @Date  : 2026/6/22 17:15
# @Author: Finks
# @Desc  :
# 装饰器工具模块
"""



"""
import time
import functools
from typing import Any, Callable, Optional, Type, Tuple

from lumilib.common.logger import logger


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_failure: Optional[Callable[[Exception, int], None]] = None):
    """
    重试装饰器：当函数执行失败时自动重试
    
    Args:
        max_retries: 最大重试次数，默认为 3
        delay: 初始重试延迟（秒），默认为 1.0
        backoff_factor: 延迟倍增因子，默认为 2.0
        exceptions: 需要重试的异常类型元组，默认为所有 Exception
        on_failure: 失败回调函数，接收异常和重试次数作为参数
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"函数 {func.__name__} 执行失败（尝试 {attempt}/{max_retries}）: {e}")
                    
                    if on_failure:
                        on_failure(e, attempt)
                    
                    if attempt < max_retries:
                        logger.info(f"等待 {current_delay:.2f} 秒后重试...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            logger.error(f"函数 {func.__name__} 已达最大重试次数 ({max_retries})，最终失败")
            raise last_exception
        
        return wrapper
    return decorator


def timing_decorator(
    log_level: str = 'info',
    log_args: bool = False,
    log_result: bool = False):
    """
    计时装饰器：记录函数执行时间
    
    Args:
        log_level: 日志级别，可选 'debug', 'info', 'warning', 'error'
        log_args: 是否记录函数参数，默认为 False
        log_result: 是否记录函数返回值，默认为 False
    
    Returns:
        装饰后的函数
    """
    level_map = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error
    }
    log_func = level_map.get(log_level.lower(), logger.info)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            if log_args:
                log_func(f"调用函数 {func.__name__}，参数: args={args}, kwargs={kwargs}")
            
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            log_message = f"函数 {func.__name__} 执行完成，耗时: {elapsed_time:.4f} 秒"
            
            if log_result:
                log_message += f"，返回值: {result}"
            
            log_func(log_message)
            
            return result
        
        return wrapper
    return decorator


def catch_exceptions(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    default_return: Any = None,
    log_error: bool = True,
    raise_error: bool = False
):
    """
    异常捕获装饰器：捕获指定异常并返回默认值
    
    Args:
        exceptions: 需要捕获的异常类型元组，默认为所有 Exception
        default_return: 异常发生时的默认返回值，默认为 None
        log_error: 是否记录错误日志，默认为 True
        raise_error: 是否重新抛出异常，默认为 False
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.error(f"函数 {func.__name__} 执行异常: {e}", exc_info=True)
                
                if raise_error:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def singleton_decorator(cls: Type) -> Type:
    """
    单例装饰器：确保类只有一个实例
    
    Args:
        cls: 要装饰的类
    
    Returns:
        单例化的类
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs) -> Any:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def memoize(cache_size: Optional[int] = None):
    """
    记忆化装饰器：缓存函数返回值
    
    Args:
        cache_size: 缓存大小限制，默认为 None（无限制）
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 创建缓存键
            key = (args, frozenset(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # 如果设置了缓存大小限制，检查并清理
            if cache_size is not None and len(cache) >= cache_size:
                # 删除最早的缓存项
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        # 添加缓存管理方法
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache)}
        
        return wrapper
    return decorator


def rate_limiter(max_calls: int = 10, time_window: float = 1.0):
    """
    限流装饰器：限制函数在指定时间窗口内的调用次数
    
    Args:
        max_calls: 最大调用次数，默认为 10
        time_window: 时间窗口（秒），默认为 1.0
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            
            # 移除时间窗口外的调用记录
            call_times[:] = [t for t in call_times if now - t < time_window]
            
            if len(call_times) >= max_calls:
                raise RuntimeError(
                    f"函数 {func.__name__} 调用过于频繁，超过限制: {max_calls}/{time_window}秒"
                )
            
            call_times.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator




if __name__ == '__main__':
    pass
    @timing_decorator('info', False, True)
    def my_fun():
        time.sleep(1)
        return 1

    my_fun()
