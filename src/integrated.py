import cv2
import mediapipe as mp
import numpy as np
import time
import pygame.midi
import os
import matplotlib.pyplot as plt
from collections import deque
from loguru import logger
from sound_generator import SoundGenerator
import pandas as pd
from datetime import datetime
import pathlib

class HandFaceSoundTracker:
    def __init__(self, camera_no=0, width=640, height=360, history_size=50):
        """
        手のランドマーク、顔の向き追跡、音生成アプリケーションの初期化
        """
        os.environ['no_proxy'] = "*"
        
        # MediaPipe設定
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # カメラ設定
        self.video_capture = cv2.VideoCapture(camera_no)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 音ジェネレーター設定
        pygame.init()
        pygame.midi.init()
        try:
            inputID, outputID = SoundGenerator.get_IOdeviceID()
            self.sound_generator = SoundGenerator(inputID=inputID, outputID=outputID)
        except Exception as e:
            logger.exception("音ジェネレーターの初期化に失敗")
            raise
        
        # 顔の向き履歴設定
        self.yaw_history = deque(maxlen=history_size)
        self.pitch_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # データ保存用のリスト
        self.data_for_export = []
        
        # セッション開始時刻とディレクトリ設定
        self.session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = pathlib.Path("output")
        self.session_dir = self.base_output_dir / self.session_start_time
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # グラフの初期化
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 手と顔の検出器
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info(f"セッションディレクトリを作成しました: {self.session_dir}")
    
    def _calculate_face_orientation(self, landmarks):
        """
        顔の向きを計算
        """
        nose_tip = landmarks.landmark[4]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        eye_distance = abs(left_eye.x - right_eye.x)
        center_x = (left_eye.x + right_eye.x) / 2
        yaw = np.arctan2(nose_tip.x - center_x, eye_distance) * 180 / np.pi
        
        eye_y = (left_eye.y + right_eye.y) / 2
        pitch = np.arctan2(nose_tip.y - eye_y, eye_distance) * 180 / np.pi
        
        return yaw, pitch
    
    def _update_orientation_plot(self):
        """
        リアルタイムでグラフを表示（保存はしない）
        """
        time_array = np.array(self.timestamps) - self.timestamps[0]
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(time_array, self.yaw_history, 'b-')
        self.ax1.set_title('Yaw (Left/Right)')
        self.ax1.set_ylabel('Angle (degrees)')
        self.ax1.grid(True)
        
        self.ax2.plot(time_array, self.pitch_history, 'r-')
        self.ax2.set_title('Pitch (Up/Down)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylabel('Angle (degrees)')
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def _create_final_plots(self):
        """
        セッション終了時に最終的なグラフを作成して保存
        """
        # 新しいフィギュアを作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # データフレームを作成
        df = pd.DataFrame(self.data_for_export, 
                         columns=['timestamp', 'yaw', 'pitch'])
        df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        # Yawのプロット
        ax1.plot(df['relative_time'], df['yaw'], 'b-', linewidth=1)
        ax1.set_title('Yaw (Left/Right) Over Time')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True)
        
        # Pitchのプロット
        ax2.plot(df['relative_time'], df['pitch'], 'r-', linewidth=1)
        ax2.set_title('Pitch (Up/Down) Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angle (degrees)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # グラフを保存
        plot_path = self.session_dir / 'orientation_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"グラフを保存しました: {plot_path}")
    
    def _save_data_to_csv(self):
        """
        記録したデータをCSVファイルに保存
        """
        if not self.data_for_export:
            logger.warning("保存するデータがありません")
            return
            
        df = pd.DataFrame(self.data_for_export, 
                         columns=['timestamp', 'yaw', 'pitch'])
        
        # 相対時間を計算
        df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        # CSVファイルに保存
        csv_path = self.session_dir / 'face_orientation.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"データを保存しました: {csv_path}")
        
        # 基本的な統計情報を保存
        stats_path = self.session_dir / 'statistics.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("基本統計情報\n")
            f.write("=" * 20 + "\n\n")
            f.write("Yaw (左右の向き):\n")
            f.write(f"平均: {df['yaw'].mean():.2f}度\n")
            f.write(f"標準偏差: {df['yaw'].std():.2f}度\n")
            f.write(f"最小値: {df['yaw'].min():.2f}度\n")
            f.write(f"最大値: {df['yaw'].max():.2f}度\n\n")
            f.write("Pitch (上下の向き):\n")
            f.write(f"平均: {df['pitch'].mean():.2f}度\n")
            f.write(f"標準偏差: {df['pitch'].std():.2f}度\n")
            f.write(f"最小値: {df['pitch'].min():.2f}度\n")
            f.write(f"最大値: {df['pitch'].max():.2f}度\n")
    
    def run(self):
        """
        メインアプリケーションループ
        """
        start_time = time.time()
        
        try:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                # フレーム前処理
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 手と顔の検出
                hands_results = self.hands.process(image_rgb)
                face_results = self.face_mesh.process(image_rgb)
                
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # 手のランドマーク処理
                if hands_results.multi_hand_landmarks:
                    for landmarks in hands_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # 音生成
                        new_notes = self.sound_generator.new_notes(
                            landmarks.landmark[9].x, 
                            landmarks.landmark[9].y
                        )
                        self.sound_generator.update_notes(new_notes)
                
                # 顔の向き処理
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    yaw, pitch = self._calculate_face_orientation(face_landmarks)
                    
                    current_time = time.time() - start_time
                    self.yaw_history.append(yaw)
                    self.pitch_history.append(pitch)
                    self.timestamps.append(current_time)
                    
                    # データを記録
                    self.data_for_export.append([time.time(), yaw, pitch])
                    
                    # 角度を画像に表示
                    cv2.putText(image, f'Yaw: {yaw:.1f}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Pitch: {pitch:.1f}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # リアルタイムグラフの更新
                    if len(self.timestamps) > 1:
                        self._update_orientation_plot()
                
                # フレーム表示
                cv2.imshow('Hand and Face Tracking', image)
                
                # 終了条件
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    break
                
                time.sleep(0.01)
        
        except Exception as e:
            logger.exception("処理中にエラーが発生")
        
        finally:
            # データとグラフの保存
            self._save_data_to_csv()
            self._create_final_plots()
            
            # クリーンアップ処理
            self.sound_generator.end()
            plt.close('all')
            self.video_capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
def main():
    """
    アプリケーション起動
    """
    try:
        tracker = HandFaceSoundTracker()
        tracker.run()
    except Exception as e:
        logger.exception("アプリケーションの起動に失敗")
    finally:
        # 最終的なクリーンアップ
        cv2.destroyAllWindows()
        plt.close('all')

if __name__ == '__main__':
    main()