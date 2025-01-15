import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import animation
from pathlib import Path
from src.models.point import Point
from loguru import logger

class DataVisualizer:
    """
    各種データを可視化するクラス
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.stop_time = 40
    
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
            ax1.axvline(x=self.stop_time, color='red', linestyle='--', linewidth=1)  # 赤い縦線
            ax1.set_title('Yaw (Left/Right) Over Time')
            ax1.set_ylabel('Angle (degrees)')
            ax1.grid(True)
            
            ax2.plot(df['relative_time'], df['pitch'], 'r-', linewidth=1)
            ax2.axvline(x=self.stop_time, color='red', linestyle='--', linewidth=1)  # 赤い縦線
            ax2.set_title('Pitch (Up/Down) Over Time')
            ax2.set_ylabel('Angle (degrees)')
            ax2.grid(True)
            
            ax3.plot(df['relative_time'], df['roll'], 'g-', linewidth=1)
            ax3.axvline(x=self.stop_time, color='red', linestyle='--', linewidth=1)  # 赤い縦線
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
            is_palm_up = []
            
            for data in hand_trajectory_data.values():
                timestamps.extend(data['timestamp'])
                x_coords.extend(data['x'])
                y_coords.extend(data['y'])
                z_coords.extend(data['z'])
                is_palm_up.extend(data['is_palm_up'])

            # データを時系列でソート
            sorted_indices = np.argsort(timestamps)
            x_coords = np.array(x_coords)[sorted_indices]
            y_coords = np.array(y_coords)[sorted_indices]
            z_coords = np.array(z_coords)[sorted_indices]
            is_palm_up = np.array(is_palm_up)[sorted_indices]
            timestamps = np.array(timestamps)[sorted_indices]

            # 3Dプロットの設定
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # プロット用のラインとポイントを作成
            full_line = ax.plot([], [], [], 
                                c='blue',
                                alpha=0.7,
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

            # 固定された視点
            ax.view_init(elev=270, azim=90)

            def update(frame):
                # 軌跡全体の更新
                color = 'blue' if is_palm_up[frame] else 'red'
                
                # 線の色を変更
                full_line.set_color(color)

                full_line.set_data(x_coords[:frame + 1],
                                y_coords[:frame + 1])
                full_line.set_3d_properties(z_coords[:frame + 1])
                
                # 現在位置の点の更新
                if frame < len(x_coords):
                    point.set_data([x_coords[frame]], 
                                [y_coords[frame]])
                    point.set_3d_properties([z_coords[frame]])
                
                return full_line, point

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

    def create_cumulative_distance(self, hand_trajectory_data):
        if not hand_trajectory_data:
            logger.warning("手の軌跡データがありません")
        else:
            try:
                dfs = []
                for hand_id, data in hand_trajectory_data.items():
                    df = pd.DataFrame(data)
                    df['hand_id'] = hand_id
                    dfs.append(df)
                
                df_hands = pd.concat(dfs, ignore_index=True)
                df_hands['relative_time'] = df_hands['timestamp'] - df_hands['timestamp'].min()

                df_hands = df_hands[df_hands['hand_id'] == 0]

                distances = [0]
                for i in range(1, len(df_hands)):
                    point_1 = Point(x = df_hands.iloc[i-1]['x'], y = df_hands.iloc[i-1]['y'], z = df_hands.iloc[i-1]['z'])
                    point_2 = Point(x = df_hands.iloc[i]['x'], y = df_hands.iloc[i]['y'], z = df_hands.iloc[i]['z'])
                    dist = point_1.distance_to(point_2)
                    distances.append(distances[-1] + dist)
                
                df_hands['culmulative_distance'] = distances

                plt.figure(figsize=(10, 6))
                plt.axvline(x=self.stop_time, color='red', linestyle='--', linewidth=1)
                plt.plot(df_hands['relative_time'], df_hands['culmulative_distance'], marker='o', label='Cumulative Distance')
                plt.title('Cumulative Distance Over Time', fontsize=14)
                plt.xlabel('Relative Time (s)', fontsize=12)
                plt.ylabel('Cumulative Distance', fontsize=12)
                plt.grid(True)
                plt.legend()

                plot_path = self.session_dir / 'cumulative_distance_plot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')

                
                logger.info(f"手の移動距離のグラフを保存しました: {plot_path}")

            except Exception as e:
                logger.error(f"グラフ作成中にエラー: {e}")


    def create_hand_speed_plot(self, hand_trajectory_data):
        if not hand_trajectory_data:
            logger.warning("手の軌跡データがありません")
        else:
            try:
                dfs = []
                for hand_id, data in hand_trajectory_data.items():
                    df = pd.DataFrame(data)
                    df['hand_id'] = hand_id
                    dfs.append(df)
                
                df_hands = pd.concat(dfs, ignore_index=True)
                df_hands['relative_time'] = df_hands['timestamp'] - df_hands['timestamp'].min()

                # hand_id == 0 のデータを抽出
                df_hands = df_hands[df_hands['hand_id'] == 0]

                speeds = [0]  # 最初の速度は 0 とする
                for i in range(1, len(df_hands)):
                    point_1 = Point(x=df_hands.iloc[i-1]['x'], y=df_hands.iloc[i-1]['y'], z=df_hands.iloc[i-1]['z'])
                    point_2 = Point(x=df_hands.iloc[i]['x'], y=df_hands.iloc[i]['y'], z=df_hands.iloc[i]['z'])
                    dist = point_1.distance_to(point_2)
                    time_diff = df_hands.iloc[i]['relative_time'] - df_hands.iloc[i-1]['relative_time']
                    speed = dist / time_diff if time_diff != 0 else 0
                    speeds.append(speed)
                
                df_hands['speed'] = speeds

                # グラフの作成
                plt.figure(figsize=(10, 6))
                plt.axvline(x=self.stop_time, color='red', linestyle='--', linewidth=1)
                plt.plot(df_hands['relative_time'], df_hands['speed'], label='Hand Speed')
                plt.title('Hand Speed Over Time', fontsize=14)
                plt.xlabel('Relative Time (s)', fontsize=12)
                plt.ylabel('Speed (units/s)', fontsize=12)
                plt.grid(True)
                plt.legend()

                plot_path = self.session_dir / 'hand_speed_plot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')

                logger.info(f"手の速度のグラフを保存しました: {plot_path}")

            except Exception as e:
                logger.error(f"グラフ作成中にエラー: {e}")

if __name__ == "__main__":
    file_path = ""
    df = pd.read_csv(file_path)