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
    C_MAJOR = ([60, 64, 67, 72, 76, 79], "C Major")  # C major
    A_MINOR = ([60, 64, 69, 72, 76, 81], "A Minor")  # A minor
    DISSONANCE = ([60, 58, 57, 56, 55, 54], "Dissonance")  # Dissonant notes
    
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
        self.volume = 64  # MIDI標準のベロシティ
        self.current_scale = Scale.C_MAJOR
        self.current_notes: Optional[List[int]] = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.goal_point = Point(0.5, 0.5)
        
        try:
            self.output = mido.open_output(output_name)
            logger.info(f"MIDI出力デバイス '{output_name}' の初期化に成功")
        except Exception as e:
            logger.error(f"MIDI出力デバイスの初期化に失敗: {e}")
            raise RuntimeError("MIDI出力デバイスの初期化に失敗しました") from e

    def end(self) -> None:
        """リソースの解放とクリーンアップ"""
        try:
            self._stop_current_notes()
            self.executor.shutdown(wait=True)
            self.output.close()
            logger.info("SoundGenerator終了処理完了")
        except Exception as e:
            logger.error(f"終了処理中にエラー: {e}")

    def set_instrument(self, program: int) -> None:
        """
        MIDI音色を変更する
        Args:
            program: MIDI program number (0-127)
        """
        try:
            self.output.send(Message('program_change', program=program))
            logger.info(f"音色を変更: program={program}")
        except Exception as e:
            logger.error(f"音色の変更中にエラー: {e}")
            
    def _play_new_notes(self, notes: List[int]) -> None:
        """新しい音符を再生"""
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
        except Exception as e:
            logger.error(f"ノート停止中にエラー: {e}")
             
    def update_notes(self, new_notes: List[int]) -> None:
        """再生中の音符を更新"""
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
         
        dist_condition = hand_point.distance_to(self.goal_point) < 0.1
        palm_condition = is_palm_up
        
        return dist_condition and palm_condition
    
    def new_notes(self, x: float, y: float, is_palm_up: bool=False) -> List[int]:
        """座標と手のひらの向きに基づいて新しい音符を生成"""
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
        sound_gen.set_instrument(40)
        time.sleep(2)
        
        logger.info("C Major スケールのテスト")
        sound_gen.set_scale(Scale.C_MAJOR)
        sound_gen.update_notes(Scale.C_MAJOR.notes)
        time.sleep(2)
        
        logger.info("A Minor スケールのテスト")
        sound_gen.set_scale(Scale.A_MINOR)
        sound_gen.update_notes(Scale.A_MINOR.notes)
        time.sleep(2)
        
        logger.info("不協和音のテスト")
        sound_gen.update_notes(Scale.DISSONANCE.notes)
        time.sleep(2)
        
        logger.info("座標ベースの音生成テスト")
        test_coordinates = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]
        for x, y in test_coordinates:
            notes = sound_gen.new_notes(x, y)
            sound_gen.update_notes(notes)
            time.sleep(1)
        
    except Exception as e:
        logger.exception(f"テスト中にエラーが発生:{e}")
    finally:
        sound_gen.end()

if __name__ == "__main__":
    test_sound_generator()
