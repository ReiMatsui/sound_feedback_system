from abc import abstractmethod
from dataclasses import Field
from pydantic import BaseModel
from typing import Any, List, Optional
import threading
from loguru import logger
import time

class Timer:
    """
    タイマー機能の基底クラス
    """
    def __init__(self, on_timer_end):
        self.timer: Optional[threading.Timer] = None
        self.duration = float('inf')  # デフォルトは無制限
        self.start_time = None      
        self.on_timer_end = on_timer_end
        
    def set_duration(self, seconds: float) -> None:
        """実験の制限時間を設定"""
        self.duration = seconds
        self.start_time = time.time()
        
        # 既存のタイマーをキャンセル
        if self.timer:
            self.timer.cancel()
        
        # 新しいタイマーを設定
        self.timer = threading.Timer(seconds, self.on_timer_end)
        self.timer.start()
        logger.info(f"実験時間を {seconds} 秒に設定")

    def cancel(self):
        self.timer.cancel()
        return 
    
    def get_remaining_time(self) -> float:
        """残り時間を取得（秒）"""
        if self.start_time is None:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.duration - elapsed)
        return remaining