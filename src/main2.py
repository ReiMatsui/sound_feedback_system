import cv2
import mediapipe as mp
import numpy as np
import time
import pygame.midi
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from collections import deque
from loguru import logger
from sound_generator import SoundGenerator
from garageband_handler import GarageBandHandler
import pandas as pd
from datetime import datetime
import pathlib

class HandFaceSoundTracker:
    def __init__(self, camera_no: int = 0, width: int = 640, height: int = 360, history_size:int = 50):
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
        
        # 動画保存の設定
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 30.0
        
        
        # 音ジェネレーター設定
        pygame.init()
        pygame.midi.init()
        try:
            input_id, output_id = SoundGenerator.get_IOdeviceID()
            self.sound_generator = SoundGenerator(input_id=input_id, output_id=output_id)
            
        except Exception as e:
            logger.exception("音ジェネレーターの初期化に失敗")
            raise
        
        # データ保存用の設定
        self.face_orientation_data = []  # 顔の向きデータ
        self.hand_trajectory_data = {}   # 手の軌跡データ（複数の手に対応）
        
        # セッション開始時刻とディレクトリ設定
        self.session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = pathlib.Path("output")
        self.session_dir = self.base_output_dir / self.session_start_time
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # 動画ライター初期化
        video_path = str(self.session_dir / 'tracking_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            self.fps, 
            (self.frame_width, self.frame_height)
        )
        
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

    def _process_hand_data(self, landmarks, hand_id):
        """
        手のランドマークデータを処理し、軌跡データを更新
        """
        landmark_9 = landmarks.landmark[9]
        timestamp = time.time()
        
        if hand_id not in self.hand_trajectory_data:
            self.hand_trajectory_data[hand_id] = {
                'timestamp': [],
                'x': [],
                'y': [],
                'z': []
            }
            
        self.hand_trajectory_data[hand_id]['timestamp'].append(timestamp)
        self.hand_trajectory_data[hand_id]['x'].append(landmark_9.x)
        self.hand_trajectory_data[hand_id]['y'].append(landmark_9.y)
        self.hand_trajectory_data[hand_id]['z'].append(landmark_9.z)

    def _create_face_orientation_plots(self):
        """
        顔の向きのグラフを作成して保存
        """
        if not self.face_orientation_data:
            logger.warning("顔の向きデータがありません")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        df = pd.DataFrame(self.face_orientation_data, 
                         columns=['timestamp', 'yaw', 'pitch'])
        df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        ax1.plot(df['relative_time'], df['yaw'], 'b-', linewidth=1)
        ax1.set_title('Yaw (Left/Right) Over Time')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True)
        
        ax2.plot(df['relative_time'], df['pitch'], 'r-', linewidth=1)
        ax2.set_title('Pitch (Up/Down) Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angle (degrees)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = self.session_dir / 'face_orientation_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"顔の向きグラフを保存しました: {plot_path}")

    def _create_3d_trajectory_animation(self):
        """
        手の軌跡の3Dアニメーションを作成して保存
        """
        if not self.hand_trajectory_data:
            logger.warning("手の軌跡データがありません")
            return

        # データフレームの作成
        dfs = []
        for hand_id, data in self.hand_trajectory_data.items():
            df = pd.DataFrame(data)
            df['id'] = hand_id
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df['relative_time'] = df['timestamp'] - df['timestamp'].min()

        # 3Dプロットの初期設定
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 各手のデータを準備
        data = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.hand_trajectory_data)))
        
        for i, hand_id in enumerate(self.hand_trajectory_data.keys()):
            hand_df = df[df['id'] == hand_id]
            data.append({
                'points': np.array([hand_df['x'], hand_df['y'], hand_df['z']]),
                'color': colors[i]
            })

        # 初期の線を作成
        lines = []
        points = []
        for hand_data in data:
            line = ax.plot([], [], [], 
                         c=hand_data['color'],
                         alpha=0.5,
                         linewidth=2)[0]
            lines.append(line)
            
            point = ax.plot([], [], [],
                          'o',
                          c=hand_data['color'],
                          markersize=8)[0]
            points.append(point)

        # 軸の範囲設定
        margin = 0.1
        ax.set_xlim([df['x'].min() - margin, df['x'].max() + margin])
        ax.set_ylim([df['y'].min() - margin, df['y'].max() + margin])
        ax.set_zlim([df['z'].min() - margin, df['z'].max() + margin])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Hand Trajectory (3D)')
        ax.grid(True)
        
        trail_length = 30

        def update(frame):
            ax.view_init(elev=20, azim=45)
            
            for i, (line, point, hand_data) in enumerate(zip(lines, points, data)):
                points_data = hand_data['points']
                
                start_idx = max(0, frame - trail_length)
                end_idx = frame + 1
                
                if end_idx > points_data.shape[1]:
                    end_idx = points_data.shape[1]
                    start_idx = max(0, end_idx - trail_length)
                
                line.set_data(points_data[0:2, start_idx:end_idx])
                line.set_3d_properties(points_data[2, start_idx:end_idx])
                
                if end_idx > 0:
                    point.set_data([points_data[0, end_idx-1]], 
                                 [points_data[1, end_idx-1]])
                    point.set_3d_properties([points_data[2, end_idx-1]])

            return lines + points

        # アニメーションの作成と保存
        num_frames = len(df['timestamp'].unique())
        ani = animation.FuncAnimation(fig, 
                                    update,
                                    frames=num_frames,
                                    interval=50,
                                    blit=True)

        animation_path = self.session_dir / 'hand_trajectory_3d.mp4'
        writer = animation.FFMpegWriter(fps=20,
                                      metadata=dict(artist='HandTracker'),
                                      bitrate=5000)
        ani.save(str(animation_path), writer=writer)
        
        plt.close(fig)
        logger.info(f"3Dアニメーションを保存しました: {animation_path}")

    def _save_data(self):
        """
        すべてのデータをCSVファイルに保存
        """
        # 顔の向きデータの保存
        if self.face_orientation_data:
            df_face = pd.DataFrame(self.face_orientation_data,
                                 columns=['timestamp', 'yaw', 'pitch'])
            df_face['relative_time'] = df_face['timestamp'] - df_face['timestamp'].iloc[0]
            df_face.to_csv(self.session_dir / 'face_orientation.csv', index=False)
            
        # 手の軌跡データの保存
        if self.hand_trajectory_data:
            dfs = []
            for hand_id, data in self.hand_trajectory_data.items():
                df = pd.DataFrame(data)
                df['hand_id'] = hand_id
                dfs.append(df)
            
            df_hands = pd.concat(dfs, ignore_index=True)
            df_hands['relative_time'] = df_hands['timestamp'] - df_hands['timestamp'].min()
            df_hands.to_csv(self.session_dir / 'hand_trajectories.csv', index=False)
        
        logger.info("すべてのデータを保存しました")

    def run(self):
        """
        メインアプリケーションループ
        """
        try:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                hands_results = self.hands.process(image_rgb)
                face_results = self.face_mesh.process(image_rgb)
                
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # 手のランドマーク処理
                if hands_results.multi_hand_landmarks:
                    for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
                        self.mp_drawing.draw_landmarks(
                            image,
                            landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # 手のデータを記録
                        self._process_hand_data(landmarks, i)
                        
                        # 音生成 (最初の手のみ)
                        if i == 0:
                            new_notes = self.sound_generator.new_notes(
                                landmarks.landmark[9].x,
                                landmarks.landmark[9].y
                            )
                            self.sound_generator.update_notes(new_notes)
                
                # 顔の向き処理
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    yaw, pitch = self._calculate_face_orientation(face_landmarks)
                    
                    # 顔の向きデータを記録
                    self.face_orientation_data.append([time.time(), yaw, pitch])
                    
                    # 角度を画像に表示
                    cv2.putText(image, f'Yaw: {yaw:.1f}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Pitch: {pitch:.1f}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # フレームを動画ファイルに書き込み
                self.video_writer.write(image)
                
                #フレーム設定
                
                # cv2.namedWindow('Hand and Face Tracking', cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty('Hand and Face Tracking',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                
                # フレーム表示
                cv2.imshow('Hand and Face Tracking', image)    
                
                # 終了条件
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESCキーまたは'q'キーで終了
                    break
                
                time.sleep(0.01)  # 少し待機してCPU負荷を下げる
        
        except Exception as e:
            logger.exception("処理中にエラーが発生")
        
        finally:
            # クリーンアップ処理
            self.sound_generator.end()
            
            time.sleep(5)
            self.video_writer.release()
            plt.close('all')
            self.video_capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            time.sleep(5)
            # すべてのデータを保存
            self._save_data()
            
            # グラフとアニメーションの生成
            self._create_face_orientation_plots()
            # self._create_3d_trajectory_animation()   

            

            
            logger.info("アプリケーションを終了しました")

def main():
    """
    アプリケーション起動
    """
    try:
        # ログの設定
        logger.add(
            "logs/app_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            encoding="utf-8"
        )
        
        logger.info("アプリケーションを開始します")
        midi_filepath = "/Users/matsuirei/Music/GarageBand/sample.band"
        midi_handler = GarageBandHandler(file_path=midi_filepath)
        midi_handler.open_file()
        time.sleep(5) # GarageBandのファイルが開くまで待機
        logger.info("GarageBandのプロジェクト準備完了")
        # トラッカーの初期化と実行
        tracker = HandFaceSoundTracker()
        tracker.run()
        
    except Exception as e:
        logger.exception("アプリケーションの起動に失敗")
    
    finally:
        # 最終的なクリーンアップ
        cv2.destroyAllWindows()
        plt.close('all')
        midi_handler.close_file()
        logger.info("アプリケーションを終了します")

if __name__ == '__main__':
    main()