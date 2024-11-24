import pygame.midi
import time
import threading
from enum import Enum
import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """別の点との距離を計算"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class Scale(Enum):
    """音階の定義"""
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
    def __init__(self, output_id: int, input_id: Optional[int] = None):
        """
        SoundGeneratorの初期化

        Args:
            output_id: MIDI出力デバイスのID
            input_id: MIDI入力デバイスのID（オプション）
        
        Raises:
            ValueError: 無効なデバイスIDが指定された場合
            RuntimeError: デバイスの初期化に失敗した場合
        """

        # 基本設定
        self.volume = 64  # MIDI標準のベロシティ
        self.current_scale = Scale.C_MAJOR
        self.current_notes: Optional[List[int]] = None
        
        # スレッド安全性の確保
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 目標点の設定（正規化座標）
        self.goal_point = Point(0.5, 0.5)
        
        # プレイヤーの初期化
        try:
            self.player = pygame.midi.Output(output_id)
            logger.info(f"MIDI出力デバイス (ID: {output_id}) の初期化に成功")
        except Exception as e:
            logger.error(f"MIDI出力デバイスの初期化に失敗: {e}")
            raise RuntimeError("MIDI出力デバイスの初期化に失敗しました") from e
        
        logger.info("SoundGenerator初期化完了")

    def end(self) -> None:
        """リソースの解放とクリーンアップ"""
        try:
            self._stop_current_notes()
            self.executor.shutdown(wait=True)
            if self.player:
                self.player.close()
            pygame.midi.quit()
            logger.info("SoundGenerator終了処理完了")
        except Exception as e:
            logger.error(f"終了処理中にエラー: {e}")

    def update_notes(self, new_notes: List[int]) -> None:
        """
        再生中の音符を更新

        Args:
            new_notes: 新しい音符のリスト
        """
        with self.lock:
            if self.current_notes == new_notes:
                return

            if self.current_notes:
                self._stop_current_notes()

            self.current_notes = new_notes
            self._play_new_notes(new_notes)

    def _play_new_notes(self, notes: List[int]) -> None:
        """
        新しい音符を再生

        Args:
            notes: 再生する音符のリスト
        """
        try:
            for i, note in enumerate(notes):
                self.player.note_on(note, self.volume, channel=i)
            logger.debug(f"ノート再生: {notes}")
        except Exception as e:
            logger.error(f"ノート再生中にエラー: {e}")

    def _stop_current_notes(self) -> None:
        """現在再生中の音符を停止"""
        if not self.current_notes:
            return
            
        try:
            for i, note in enumerate(self.current_notes):
                self.player.note_off(note, self.volume, channel=i)
            logger.debug(f"ノート停止: {self.current_notes}")
        except Exception as e:
            logger.error(f"ノート停止中にエラー: {e}")

    def new_notes(self, x: float, y: float) -> List[int]:
        """
        座標に基づいて新しい音符を生成

        Args:
            x: x座標（0-1）
            y: y座標（0-1）

        Returns:
            生成された音符のリスト
        """
        current_point = Point(x, y)
        dist = current_point.distance_to(self.goal_point)
        
        if dist < 0.1:
            return self.current_scale.C_MAJOR.notes
        else:
            # 距離に応じて音を変化させる
            base_note = 60  # Middle C
            note_offset = min(int(dist * 20), 24)  # 最大2オクターブまで下げる
            return [base_note - note_offset]

    def set_scale(self, scale: Scale) -> None:
        """
        使用する音階を設定

        Args:
            scale: 使用する音階
        """
        self.current_scale = scale
        logger.info(f"音階を変更: {scale.description}")

    def set_volume(self, volume: int) -> None:
        """
        音量を設定

        Args:
            volume: 音量（0-127）
        """
        self.volume = max(0, min(127, volume))
        logger.info(f"音量を設定: {self.volume}")

    def set_goal_point(self, x: float, y: float) -> None:
        """
        目標点を設定

        Args:
            x: x座標（0-1）
            y: y座標（0-1）
        """
        self.goal_point = Point(x, y)
        logger.debug(f"目標点を設定: ({x}, {y})")

    def end(self) -> None:
        """リソースの解放とクリーンアップ"""
        try:
            self._stop_current_notes()
            self.executor.shutdown(wait=True)
            if self.player:
                self.player.close()
            pygame.midi.quit()
            logger.info("SoundGenerator終了処理完了")
        except Exception as e:
            logger.error(f"終了処理中にエラー: {e}")

    @staticmethod
    def get_IOdeviceID() -> Tuple[Optional[int], int]:
        """
        利用可能なMIDIデバイスを検索し、適切な入出力デバイスIDを返す

        Returns:
            Tuple[Optional[int], int]: (入力デバイスID, 出力デバイスID)

        Raises:
            ValueError: 適切なMIDIデバイスが見つからない場合
        """
        if not pygame.midi.get_count():
            raise ValueError("MIDIデバイスが見つかりません")

        devices = []
        default_output = pygame.midi.get_default_output_id()

        # デバイス情報の収集
        for i in range(pygame.midi.get_count()):
            try:
                info = pygame.midi.get_device_info(i)
                if info is None:
                    continue

                interface, name, is_input, is_output, is_opened = info
                device_info = {
                    'id': i,
                    'interface': interface.decode('utf-8'),
                    'name': name.decode('utf-8'),
                    'is_input': bool(is_input),
                    'is_output': bool(is_output),
                    'is_opened': bool(is_opened)
                }
                devices.append(device_info)
                
                logger.info(
                    f"デバイス {i}: {device_info['name']} "
                    f"({'入力' if device_info['is_input'] else '出力'}) "
                    f"[{'使用中' if device_info['is_opened'] else '利用可能'}]"
                )
            
            except Exception as e:
                logger.warning(f"デバイス {i} の情報取得に失敗: {e}")

        # 出力デバイスの選択
        output_id = default_output if default_output >= 0 else None
        if output_id is None:
            for dev in devices:
                if dev['is_output'] and not dev['is_opened']:
                    output_id = dev['id']
                    break

        if output_id is None:
            raise ValueError("利用可能な出力デバイスが見つかりません")

        # 入力デバイスの選択（オプション）
        input_id = None
        for dev in devices:
            if dev['is_input'] and not dev['is_opened']:
                input_id = dev['id']
                break

        logger.info(f"選択された出力デバイス ID: {output_id}")
        if input_id is not None:
            logger.info(f"選択された入力デバイス ID: {input_id}")

        return input_id, output_id

def test_sound_generator():
    """SoundGeneratorのテスト関数"""
    pygame.init()
    pygame.midi.init()
    
    try:
        input_id, output_id = SoundGenerator.get_IOdeviceID()
        sound_gen = SoundGenerator(output_id, input_id)
        
        # 基本機能のテスト
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
        
        # 座標に基づく音生成のテスト
        logger.info("座標ベースの音生成テスト")
        test_coordinates = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]
        for x, y in test_coordinates:
            notes = sound_gen.new_notes(x, y)
            sound_gen.update_notes(notes)
            time.sleep(1)
        
    except Exception as e:
        logger.exception("テスト中にエラーが発生")
    finally:
        sound_gen.end()

if __name__ == "__main__":
    test_sound_generator()