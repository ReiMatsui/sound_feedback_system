import pandas as pd
from pathlib import Path
import time
import cv2
from loguru import logger
from src.utils.data_visualizer import DataVisualizer

class DataRecorder:
    """ 
    各種データを保存するクラス
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.face_orientation_data = []
        self.face_image_data = []
        self.hand_trajectory_data = {}
        self.data_visualizer = DataVisualizer(self.session_dir)
        # 画像保存用のディレクトリを作成
        self.image_dir = self.session_dir / 'images'
        self.image_dir.mkdir(exist_ok=True)
    
    def record_face_orientation(self, yaw, pitch, roll):
        """
        顔の向き情報をリストに保存
        """
        self.face_orientation_data.append([time.time(), yaw, pitch, roll])

    def record_face_image(self, face_image):
        """顔向きが大きく変化したときの画像をリストに保存"""
        timestamp = time.time()
        
        # 相対時間を計算（最初のタイムスタンプとの差）
        if self.face_orientation_data:
            first_timestamp = self.face_orientation_data[0][0]
            relative_time = timestamp - first_timestamp
        else:
            relative_time = 0
        
        # 画像データをリストに追加
        self.face_image_data.append([relative_time, face_image])
    
    
    def record_hand_trajectory(self, landmarks, hand_id):
        """ 
        手の位置を辞書に保存
        """
        if hand_id not in self.hand_trajectory_data:
            self.hand_trajectory_data[hand_id] = {'timestamp': [], 'x': [], 'y': [], 'z': []}
        
        landmark_9 = landmarks.landmark[9]
        timestamp = time.time()
        
        self.hand_trajectory_data[hand_id]['timestamp'].append(timestamp)
        self.hand_trajectory_data[hand_id]['x'].append(landmark_9.x)
        self.hand_trajectory_data[hand_id]['y'].append(landmark_9.y)
        self.hand_trajectory_data[hand_id]['z'].append(landmark_9.z)
    
    def save_data(self):
        """
        すべてのデータをCSVファイルに保存
        """
        try:
            # 顔の向きデータの保存
            if self.face_orientation_data:
                df_face = pd.DataFrame(self.face_orientation_data,
                                    columns=['timestamp', 'yaw', 'pitch', 'roll'])
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
            
            # 顔画像の保存
            if self.face_image_data:
                for relative_time, face_image in self.face_image_data:
                    # 画像のファイル名に相対時間を使う
                    image_filename = f"{relative_time:.2f}.png"
                    image_path = self.image_dir / image_filename
                    
                    # 画像を保存
                    cv2.imwrite(str(image_path), face_image)
                    logger.info(f"顔画像を保存しました: {image_path}")

            logger.info("csvデータと画像を保存しました")
        except Exception as e:
            logger.error(f"データ保存中にエラー: {e}")
    
    def visualize_data(self):
        """
        各種データを可視化して保存
        """
        self.data_visualizer.create_face_orientation_plots(self.face_orientation_data)
        self.data_visualizer.create_cumulative_distance(self.hand_trajectory_data)
        self.data_visualizer.create_hand_speed_plot(self.hand_trajectory_data)
        self.data_visualizer.create_3d_trajectory_animation(self.hand_trajectory_data)