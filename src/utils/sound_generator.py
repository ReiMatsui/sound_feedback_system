import time
import threading
from enum import Enum
import math
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import mido
from mido import Message

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """
        別の点との距離を計算
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class Scale(Enum):
    """
    音階の定義
    """
    C_MAJOR = ([60, 64, 67, 72, 76, 79], "C Major")
    A_MINOR = ([60, 64, 69, 72, 76, 81], "A Minor")
    DISSONANCE = ([60, 58, 57, 56, 55, 54], "Dissonance")
    
    def __init__(self, notes: List[int], description: str):
        self.notes = notes
        self.description = description

class SoundGenerator:
    """
    ハンドトラッキングに基づいて音を生成するクラス
    """
    def __init__(self, output_name: str):
        """
        SoundGeneratorの初期化

        Args:
            output_name: MIDI出力デバイスの名前
        
        Raises:
            RuntimeError: デバイスの初期化に失敗した場合
        """
        self.volume = 64
        self.current_scale = Scale.C_MAJOR
        self.current_notes: Optional[List[int]] = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.goal_point = Point(0.5, 0.5)
        
        # タイマー関連の新しい属性
        self.timer: Optional[threading.Timer] = None
        self.changeable_timer: Optional[threading.Timer] = None
        self.is_active = True
        self.is_changeable = True
        self.duration = float('inf')  # デフォルトは無制限
        self.changeable_duration = float('inf')
        self.start_time = None
        self.start_changeable_time = None
        
        
        try:
            self.output = mido.open_output(output_name)
            logger.info(f"MIDI出力デバイス '{output_name}' の初期化に成功")
        except Exception as e:
            logger.error(f"MIDI出力デバイスの初期化に失敗: {e}")
            raise RuntimeError("MIDI出力デバイスの初期化に失敗しました") from e

    def set_duration(self, seconds: float) -> None:
        """実験の制限時間を設定"""
        self.is_active = True
        self.duration = seconds
        self.start_time = time.time()
        
        # 既存のタイマーをキャンセル
        if self.timer:
            self.timer.cancel()
        
        # 新しいタイマーを設定
        self.timer = threading.Timer(seconds, self.stop_sound)
        self.timer.start()
        logger.info(f"実験時間を {seconds} 秒に設定")

    def stop_sound(self) -> None:
        """実験を停止し、全ての音を止める"""
        self.is_active = False
        self._stop_current_notes()
        logger.info("実験時間終了、音を停止")
        
    def get_remaining_time(self) -> float:
        """残り時間を取得（秒）"""
        if self.start_time is None:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.duration - elapsed)
        return remaining

    def set_changeable_duration(self, seconds: float) -> None:
        """実験の制限時間を設定"""
        self.is_changeable = True
        self.changeable_duration = seconds
        self.start_changeable_time = time.time()
        
        # 既存のタイマーをキャンセル
        if self.changeable_timer:
            self.changeable_timer.cancel()
        
        # 新しいタイマーを設定
        self.changeable_timer = threading.Timer(seconds, self.stop_change_sound)
        self.changeable_timer.start()
        logger.info(f"実験時間を {seconds} 秒に設定")

    def stop_change_sound(self) -> None:
        """実験を停止し、全ての音を止める"""
        self.is_changeable = False
        logger.info("実験時間終了、音を停止")
        
    def get_changeable_time(self) -> float:
        """残り時間を取得（秒）"""
        if self.start_changeable_time is None:
            return float('inf')
        
        elapsed = time.time() - self.start_changeable_time
        remaining = max(0, self.changeable_duration - elapsed)
        return remaining

    def end(self) -> None:
        """リソースの解放とクリーンアップ"""
        try:
            if self.timer:
                self.timer.cancel()
            self.is_active = False
            self._stop_current_notes()
            self.executor.shutdown(wait=True)
            self.output.close()
            logger.info("SoundGenerator終了処理完了")
        except Exception as e:
            logger.error(f"終了処理中にエラー: {e}")
            
    def _play_new_notes(self, notes: List[int]) -> None:
        """新しい音符を再生"""
        if not self.is_active:
            return
        try:
            for note in notes:
                self.output.send(Message('note_on', note=note, velocity=self.volume))
        except Exception as e:
            logger.error(f"ノート再生中にエラー: {e}")

    def _stop_current_notes(self) -> None:
        """現在再生中の音符を停止"""
        if not self.current_notes:
            return
        try:
            for note in self.current_notes:
                self.output.send(Message('note_off', note=note, velocity=self.volume))
            self.current_notes = None
        except Exception as e:
            logger.error(f"ノート停止中にエラー: {e}")
             
    def update_notes(self, new_notes: List[int]) -> None:
        """再生中の音符を更新"""
        if not self.is_active:
            return
            
        if not self.is_changeable:
            with self.lock:
                self._play_new_notes(self.current_notes)
                return
            
        with self.lock:
            if self.current_notes == new_notes:
                return

            if self.current_notes:
                self._stop_current_notes()

            self.current_notes = new_notes
            self._play_new_notes(new_notes)

    def should_play_consonant(self, hand_point: Point, is_palm_up: bool) -> bool:
        """
        協和音を再生すべきかどうかを判定
        
        条件:
        1. 手が目標点の近く (distance < 0.1)
        2. 手のひらが上を向いている
        """
        if not self.is_active:
            return False
            
        dist_condition = hand_point.distance_to(self.goal_point) < 0.1
        palm_condition = is_palm_up
        return dist_condition and palm_condition
    
    def new_notes(self, x: float, y: float, is_palm_up: bool=False) -> List[int]:
        """座標と手のひらの向きに基づいて新しい音符を生成"""
        if not self.is_active:
            return []
            
        current_point = Point(x, y)
        self.volume = 64
        
        if self.should_play_consonant(current_point, is_palm_up):
            return self.current_scale.C_MAJOR.notes
        else:
            if is_palm_up:
                self.volume = 100
            base_note = 60
            dist = current_point.distance_to(self.goal_point)
            note_offset = min(int(dist * 20), 24)
            return [base_note - note_offset]

    def set_scale(self, scale: Scale) -> None:
        """使用する音階を設定"""
        self.current_scale = scale
        logger.info(f"音階を変更: {scale.description}")

    def set_volume(self, volume: int) -> None:
        """音量を設定"""
        self.volume = max(0, min(127, volume))
        logger.info(f"音量を設定: {self.volume}")

    def set_goal_point(self, x: float, y: float) -> None:
        """目標点を設定"""
        self.goal_point = Point(x, y)
        logger.debug(f"目標点を設定: ({x}, {y})")

    @staticmethod
    def get_output_names() -> List[str]:
        """利用可能なMIDI出力デバイス名を取得"""
        return mido.get_output_names()

def test_sound_generator():
    """SoundGeneratorのテスト関数"""
    try:
        output_names = SoundGenerator.get_output_names()
        if not output_names:
            raise ValueError("利用可能なMIDI出力デバイスが見つかりません")
        logger.info(f"利用可能なMIDI出力デバイス: {output_names}")
        
        sound_gen = SoundGenerator(output_names[0])
        # 10秒の制限時間を設定
        sound_gen.set_duration(2.0)
        
        logger.info("C Major スケールのテスト")
        sound_gen.set_scale(Scale.C_MAJOR)
        sound_gen.update_notes(Scale.C_MAJOR.notes)
        time.sleep(2)
        
        logger.info(f"残り時間: {sound_gen.get_remaining_time():.1f}秒")
        
        logger.info("A Minor スケールのテスト")
        sound_gen.set_scale(Scale.A_MINOR)
        sound_gen.update_notes(Scale.A_MINOR.notes)
        time.sleep(2)
        
        
        logger.info(f"残り時間: {sound_gen.get_remaining_time():.1f}秒")
        
        logger.info("不協和音のテスト")
        sound_gen.update_notes(Scale.DISSONANCE.notes)
        time.sleep(2)
        
        logger.info("座標ベースの音生成テスト")
        test_coordinates = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]
        for x, y in test_coordinates:
            if sound_gen.is_active:  # 時間切れでないことを確認
                notes = sound_gen.new_notes(x, y)
                sound_gen.update_notes(notes)
                time.sleep(1)
                logger.info(f"残り時間: {sound_gen.get_remaining_time():.1f}秒")
        
    except Exception as e:
        logger.exception(f"テスト中にエラーが発生:{e}")
    finally:
        sound_gen.end()

if __name__ == "__main__":
    test_sound_generator()