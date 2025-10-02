"""
动态模型加载管理器
支持延迟加载和自动卸载，适用于低频高并发场景
"""

import asyncio
import gc
import logging
import time
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class LazyModelManager:
    """
    延迟加载模型管理器
    
    特性：
    1. 首次请求时才加载模型
    2. 空闲超时后自动卸载释放内存
    3. 线程安全
    4. 支持预热（提前加载）
    """
    
    def __init__(
        self,
        name: str,
        load_func: Callable,
        idle_timeout: int = 300,
        enable_auto_unload: bool = True
    ):
        """
        Args:
            name: 模型名称（用于日志）
            load_func: 模型加载函数（同步函数）
            idle_timeout: 空闲超时时间（秒），默认 5 分钟
            enable_auto_unload: 是否启用自动卸载
        """
        self.name = name
        self.load_func = load_func
        self.idle_timeout = idle_timeout
        self.enable_auto_unload = enable_auto_unload
        
        self.model = None
        self.lock = asyncio.Lock()
        self.last_access = None
        self.unload_task: Optional[asyncio.Task] = None
        self.load_count = 0
        self.unload_count = 0
    
    async def get_model(self):
        """
        获取模型实例
        首次调用时会加载模型，后续调用直接返回缓存的模型
        
        Returns:
            模型实例
        """
        async with self.lock:
            if self.model is None:
                await self._load_model()
            
            self.last_access = time.time()
            
            # 重新调度卸载任务
            if self.enable_auto_unload:
                self._schedule_unload()
            
            return self.model
    
    async def _load_model(self):
        """内部方法：加载模型"""
        logger.info(f"⏳ [{self.name}] 开始加载模型...")
        start_time = time.time()
        
        try:
            # 在线程池中执行同步加载函数
            self.model = await asyncio.to_thread(self.load_func)
            self.load_count += 1
            elapsed = time.time() - start_time
            logger.info(f"✅ [{self.name}] 模型加载完成 (耗时: {elapsed:.1f}s, 总加载次数: {self.load_count})")
        except Exception as e:
            logger.exception(f"❌ [{self.name}] 模型加载失败: {e}")
            raise
    
    def _schedule_unload(self):
        """调度自动卸载任务"""
        # 取消之前的卸载任务
        if self.unload_task and not self.unload_task.done():
            self.unload_task.cancel()
        
        # 创建新的卸载任务
        self.unload_task = asyncio.create_task(self._auto_unload())
    
    async def _auto_unload(self):
        """自动卸载任务"""
        try:
            await asyncio.sleep(self.idle_timeout)
            
            async with self.lock:
                # 再次检查是否超时（避免竞态条件）
                if self.last_access and time.time() - self.last_access >= self.idle_timeout:
                    await self._unload_model()
        except asyncio.CancelledError:
            # 任务被取消（有新请求），正常情况
            pass
        except Exception as e:
            logger.exception(f"❌ [{self.name}] 自动卸载任务异常: {e}")
    
    async def _unload_model(self):
        """内部方法：卸载模型"""
        if self.model is None:
            return
        
        logger.info(f"🗑️ [{self.name}] 开始卸载模型（空闲超时）...")
        start_time = time.time()
        
        try:
            # 删除模型引用
            del self.model
            self.model = None
            self.unload_count += 1
            
            # 强制垃圾回收
            gc.collect()
            
            # 如果使用 CUDA，清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ [{self.name}] 模型卸载完成 (耗时: {elapsed:.1f}s, 总卸载次数: {self.unload_count})")
        except Exception as e:
            logger.exception(f"❌ [{self.name}] 模型卸载失败: {e}")
    
    async def force_unload(self):
        """强制卸载模型（外部调用）"""
        async with self.lock:
            if self.unload_task and not self.unload_task.done():
                self.unload_task.cancel()
            await self._unload_model()
    
    async def warmup(self):
        """预热：提前加载模型"""
        logger.info(f"🔥 [{self.name}] 开始预热...")
        await self.get_model()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "name": self.name,
            "is_loaded": self.is_loaded(),
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "last_access": self.last_access,
            "idle_timeout": self.idle_timeout,
            "auto_unload_enabled": self.enable_auto_unload,
        }
