"""
日志工具函数 - 用于设置日志记录器
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(name: Optional[str] = None, 
                 level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 console: bool = True,
                 file_mode: str = 'a') -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        console: 是否输出到控制台
        file_mode: 文件模式，'a'表示追加，'w'表示覆盖
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """
    日志记录器混入类，为类添加日志功能
    """
    
    def __init__(self, logger_name: Optional[str] = None, level: int = logging.INFO):
        """
        初始化日志记录器混入类
        
        Args:
            logger_name: 日志记录器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)
        self.logger.setLevel(level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 设置日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 添加控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def set_log_file(self, log_file: str, file_mode: str = 'a') -> None:
        """
        设置日志文件
        
        Args:
            log_file: 日志文件路径
            file_mode: 文件模式，'a'表示追加，'w'表示覆盖
        """
        # 检查是否已经有文件处理器
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
        
        # 添加文件处理器
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)