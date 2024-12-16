import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import animation
from pathlib import Path
from loguru import logger

class DataVisualizer:
    """
    各種データを可視化するクラス
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
    
    def create_face_orientation_plots(self, face_orientation_data):
        """
        顔の向きのグラフを作成して保存
        """
        if not face_orientation_data:
            logger.warning("顔の向きデータがありません")
            return
            
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            df = pd.DataFrame(face_orientation_data, 
                             columns=['timestamp', 'yaw', 'pitch', 'roll'])
            df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
            
            ax1.plot(df['relative_time'], df['yaw'], 'b-', linewidth=1)
            ax1.set_title('Yaw (Left/Right) Over Time')
            ax1.set_ylabel('Angle (degrees)')
            ax1.grid(True)
            
            ax2.plot(df['relative_time'], df['pitch'], 'r-', linewidth=1)
            ax2.set_title('Pitch (Up/Down) Over Time')
            ax2.set_ylabel('Angle (degrees)')
            ax2.grid(True)
            
            ax3.plot(df['relative_time'], df['roll'], 'g-', linewidth=1)
            ax3.set_title('Roll (Head Tilt) Over Time')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Angle (degrees)')
            ax3.grid(True)
            
            plt.tight_layout()
            
            plot_path = self.session_dir / 'face_orientation_plot.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"顔の向きグラフを保存しました: {plot_path}")
        except Exception as e:
            logger.error(f"グラフ作成中にエラー: {e}")
    
    def create_3d_trajectory_animation(self, hand_trajectory_data):
        """
        手の軌跡の3Dアニメーションを作成して保存
        """
        if not hand_trajectory_data:
            logger.warning("手の軌跡データがありません")
            return

        try:
            # データを単純な配列に変換
            timestamps = []
            x_coords = []
            y_coords = []
            z_coords = []
            
            for data in hand_trajectory_data.values():
                timestamps.extend(data['timestamp'])
                x_coords.extend(data['x'])
                y_coords.extend(data['y'])
                z_coords.extend(data['z'])

            # データを時系列でソート
            sorted_indices = np.argsort(timestamps)
            x_coords = np.array(x_coords)[sorted_indices]
            y_coords = np.array(y_coords)[sorted_indices]
            z_coords = np.array(z_coords)[sorted_indices]
            timestamps = np.array(timestamps)[sorted_indices]

            # 3Dプロットの設定
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # プロット用のラインとポイントを作成
            line = ax.plot([], [], [], 
                        c='blue',
                        alpha=0.5,
                        linewidth=2)[0]
            point = ax.plot([], [], [],
                        'o',
                        c='red',
                        markersize=8)[0]

            margin = 0.1
            ax.set_xlim([min(x_coords) - margin, max(x_coords) + margin])
            ax.set_ylim([min(y_coords) - margin, max(y_coords) + margin])
            ax.set_zlim([min(z_coords) - margin, max(z_coords) + margin])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Hand Trajectory (3D)')
            ax.grid(True)
            
            trail_length = 30

            def update(frame):
                ax.view_init(elev=20, azim=frame)
                
                start_idx = max(0, frame - trail_length)
                end_idx = frame + 1
                
                if end_idx > len(x_coords):
                    end_idx = len(x_coords)
                    start_idx = max(0, end_idx - trail_length)
                
                # 軌跡の更新
                line.set_data(x_coords[start_idx:end_idx],
                            y_coords[start_idx:end_idx])
                line.set_3d_properties(z_coords[start_idx:end_idx])
                
                # 現在位置の点の更新
                if end_idx > 0:
                    point.set_data([x_coords[end_idx-1]], 
                                [y_coords[end_idx-1]])
                    point.set_3d_properties([z_coords[end_idx-1]])
                
                return line, point

            # アニメーションの作成と保存
            num_frames = len(timestamps)
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
        except Exception as e:
            logger.error(f"3Dアニメーション作成中にエラー: {e}")
