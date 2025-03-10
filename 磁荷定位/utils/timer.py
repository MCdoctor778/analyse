"""
计时器类 - 用于测量代码执行时间
"""

import time
from typing import Dict, List, Optional
from contextlib import contextmanager


class Timer:
    """
    计时器类，用于测量代码执行时间
    """
    
    def __init__(self):
        """
        初始化计时器
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.running = False
        
        # 用于记录多个时间段
        self.records = {}
    
    def start(self) -> None:
        """
        开始计时
        """
        self.start_time = time.time()
        self.running = True
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            经过的时间（秒）
        """
        if not self.running:
            raise RuntimeError("计时器未启动")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.running = False
        
        return self.elapsed_time
    
    def reset(self) -> None:
        """
        重置计时器
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.running = False
    
    def get_elapsed_time(self) -> float:
        """
        获取经过的时间
        
        Returns:
            经过的时间（秒）
        """
        if self.running:
            return time.time() - self.start_time
        elif self.elapsed_time is not None:
            return self.elapsed_time
        else:
            raise RuntimeError("计时器未启动")
    
    def record(self, name: str) -> None:
        """
        记录当前时间点
        
        Args:
            name: 记录名称
        """
        if not self.running:
            raise RuntimeError("计时器未启动")
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if name in self.records:
            self.records[name].append(elapsed)
        else:
            self.records[name] = [elapsed]
    
    def get_record(self, name: str) -> List[float]:
        """
        获取记录
        
        Args:
            name: 记录名称
            
        Returns:
            记录的时间点列表
        """
        if name not in self.records:
            raise KeyError(f"记录 '{name}' 不存在")
        
        return self.records[name]
    
    def get_all_records(self) -> Dict[str, List[float]]:
        """
        获取所有记录
        
        Returns:
            所有记录
        """
        return self.records
    
    def clear_records(self) -> None:
        """
        清除所有记录
        """
        self.records = {}
    
    def __str__(self) -> str:
        """
        转换为字符串
        
        Returns:
            计时器状态字符串
        """
        if self.running:
            return f"Timer(running, elapsed: {self.get_elapsed_time():.6f}s)"
        elif self.elapsed_time is not None:
            return f"Timer(stopped, elapsed: {self.elapsed_time:.6f}s)"
        else:
            return "Timer(reset)"
    
    @contextmanager
    def time_block(self, name: Optional[str] = None):
        """
        计时代码块
        
        Args:
            name: 记录名称
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if name is not None:
                if name in self.records:
                    self.records[name].append(elapsed)
                else:
                    self.records[name] = [elapsed]
            print(f"代码块执行时间: {elapsed:.6f}秒")


# 创建一个全局计时器实例
timer = Timer()


@contextmanager
def time_this(name: Optional[str] = None):
    """
    计时代码块的便捷函数
    
    Args:
        name: 记录名称
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if name:
            print(f"{name} 执行时间: {elapsed:.6f}秒")
        else:
            print(f"代码块执行时间: {elapsed:.6f}秒")