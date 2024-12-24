import pandas as pd
from pathlib import Path
import time
from loguru import logger
from src.utils.data_visualizer import DataVisualizer

class DataRecorder:
    """ 
    各種データを保存するクラス
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.face_orientation_data = []
        self.hand_trajectory_data = {}
        self.data_visualizer = DataVisualizer(self.session_dir)
    
    def record_face_orientation(self, yaw, pitch, roll):
        """
        顔の向き情報をリストに保存
        """
        self.face_orientation_data.append([time.time(), yaw, pitch, roll])
    
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
            
            logger.info("csvデータを保存しました")
        except Exception as e:
            logger.error(f"データ保存中にエラー: {e}")
    
    def visualize_data(self):
        """
        各種データを可視化して保存
        """
        self.data_visualizer.create_face_orientation_plots(self.face_orientation_data)
        self.data_visualizer.create_cumulative_distance(self.hand_trajectory_data)
        # self.data_visualizer.create_3d_trajectory_animation(self.hand_trajectory_data)