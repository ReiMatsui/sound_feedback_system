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
        palm upの状態に応じて軌跡の色を変更し、直近30フレームの分散を表示
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

            start_time = timestamps[0]

            # 分散計算用の関数
            def calculate_variance(data, current_frame, window_size=30):
                start_idx = max(0, current_frame - window_size + 1)
                window_data = data[start_idx:current_frame + 1]
                if len(window_data) > 0:
                    return np.var(window_data)
                return 0.0

            # figureとグリッドの設定
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

            # 3Dプロット用のサブプロット
            ax = fig.add_subplot(gs[0], projection='3d')
            
            # テキスト表示用のサブプロット
            text_ax = fig.add_subplot(gs[1])
            text_ax.axis('off')

            # フォントサイズを設定
            TITLE_SIZE = 16
            LABEL_SIZE = 14
            INFO_SIZE = 12
            LEGEND_SIZE = 12

            # プロットのタイトルとラベルのフォントサイズを設定
            ax.set_title('Hand Trajectory (3D)', fontsize=TITLE_SIZE)
            ax.set_xlabel('X', fontsize=LABEL_SIZE)
            ax.set_ylabel('Y', fontsize=LABEL_SIZE)
            ax.set_zlabel('Z', fontsize=LABEL_SIZE)

            # テキストオブジェクトを作成
            time_text = text_ax.text(0.05, 0.90, '', 
                                transform=text_ax.transAxes, 
                                fontsize=INFO_SIZE,
                                fontweight='bold')
            palm_state_text = text_ax.text(0.05, 0.85, '', 
                                        transform=text_ax.transAxes, 
                                        fontsize=INFO_SIZE,
                                        fontweight='bold')
            coord_text = text_ax.text(0.05, 0.7, '', 
                                    transform=text_ax.transAxes, 
                                    fontsize=INFO_SIZE,
                                    fontweight='bold')
            
            # 分散表示用のテキスト
            variance_title = text_ax.text(0.05, 0.65, 'Variance (last 30 frames):',
                                        transform=text_ax.transAxes,
                                        fontsize=INFO_SIZE,
                                        fontweight='bold')
            x_var_text = text_ax.text(0.05, 0.60, '',
                                    transform=text_ax.transAxes,
                                    fontsize=INFO_SIZE)
            y_var_text = text_ax.text(0.05, 0.55, '',
                                    transform=text_ax.transAxes,
                                    fontsize=INFO_SIZE)
            
            # 凡例用のテキスト
            text_ax.text(0.05, 0.4, 'Blue: Palm Up', 
                        transform=text_ax.transAxes, 
                        color='blue',
                        fontsize=LEGEND_SIZE)
            text_ax.text(0.05, 0.3, 'Red: Palm Down', 
                        transform=text_ax.transAxes, 
                        color='red',
                        fontsize=LEGEND_SIZE)

            # 軸の目盛りのフォントサイズを設定
            ax.tick_params(axis='x', labelsize=LABEL_SIZE)
            ax.tick_params(axis='y', labelsize=LABEL_SIZE)
            ax.tick_params(axis='z', labelsize=LABEL_SIZE)

            palm_up_lines = []
            palm_down_lines = []
            point = ax.plot([], [], [], 'o', c='red', markersize=8)[0]

            margin = 0.1
            ax.set_xlim([min(x_coords) - margin, max(x_coords) + margin])
            ax.set_ylim([min(y_coords) - margin, max(y_coords) + margin])
            ax.set_zlim([min(z_coords) - margin, max(z_coords) + margin])
            
            ax.grid(True)
            ax.view_init(elev=270, azim=90)

            def update(frame):
                # 以前のラインをクリア
                for line in palm_up_lines + palm_down_lines:
                    line.remove()
                palm_up_lines.clear()
                palm_down_lines.clear()
                
                # フレームまでのデータを分割して描画
                current_palm_up_x = []
                current_palm_up_y = []
                current_palm_up_z = []
                current_palm_down_x = []
                current_palm_down_y = []
                current_palm_down_z = []
                
                for i in range(frame + 1):
                    if i > 0:
                        if is_palm_up[i]:
                            if not current_palm_up_x:
                                current_palm_up_x.extend([x_coords[i-1], x_coords[i]])
                                current_palm_up_y.extend([y_coords[i-1], y_coords[i]])
                                current_palm_up_z.extend([z_coords[i-1], z_coords[i]])
                            else:
                                current_palm_up_x.append(x_coords[i])
                                current_palm_up_y.append(y_coords[i])
                                current_palm_up_z.append(z_coords[i])
                        else:
                            if not current_palm_down_x:
                                current_palm_down_x.extend([x_coords[i-1], x_coords[i]])
                                current_palm_down_y.extend([y_coords[i-1], y_coords[i]])
                                current_palm_down_z.extend([z_coords[i-1], z_coords[i]])
                            else:
                                current_palm_down_x.append(x_coords[i])
                                current_palm_down_y.append(y_coords[i])
                                current_palm_down_z.append(z_coords[i])
                
                # palm upの軌跡を青色で描画
                if current_palm_up_x:
                    line_up = ax.plot(current_palm_up_x, 
                                    current_palm_up_y, 
                                    current_palm_up_z,
                                    c='blue',
                                    alpha=0.7,
                                    linewidth=2)[0]
                    palm_up_lines.append(line_up)
                
                # palm downの軌跡を赤色で描画
                if current_palm_down_x:
                    line_down = ax.plot(current_palm_down_x,
                                    current_palm_down_y,
                                    current_palm_down_z,
                                    c='red',
                                    alpha=0.7,
                                    linewidth=2)[0]
                    palm_down_lines.append(line_down)
                
                # 現在位置の点と情報を更新
                if frame < len(x_coords):
                    point.set_data([x_coords[frame]], [y_coords[frame]])
                    point.set_3d_properties([z_coords[frame]])
                    
                    relative_time = timestamps[frame] - start_time

                    # テキスト情報を更新
                    time_text.set_text(f'Time: {relative_time:.2f} sec')
                    palm_state_text.set_text(f'Palm State: {"UP" if is_palm_up[frame] else "DOWN"}')
                    coord_text.set_text(f'Position:\nX: {x_coords[frame]:.2f}\nY: {y_coords[frame]:.2f}\nZ: {z_coords[frame]:.2f}')
                    
                    # 分散を計算して表示
                    x_variance = calculate_variance(x_coords, frame)
                    y_variance = calculate_variance(y_coords, frame)
                    x_var_text.set_text(f'X variance: {x_variance:.6f}')
                    y_var_text.set_text(f'Y variance: {y_variance:.6f}')
                
                return palm_up_lines + palm_down_lines + [point, time_text, palm_state_text, coord_text,
                                                        x_var_text, y_var_text]

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

    def plot_trajectory_variance(self, hand_trajectory_data, window_size=30, save_path="variance_plot.png"):
        """
        手の軌跡データの分散を計算してグラフ化する関数
        
        Parameters:
        -----------
        hand_trajectory_data : dict
            手の軌跡データを含む辞書
        window_size : int
            分散を計算する際のウィンドウサイズ（デフォルト: 30フレーム）
        save_path : str or Path, optional
            グラフを保存するパス。Noneの場合は保存しない
        """
        try:
            # データの準備
            timestamps = []
            x_coords = []
            y_coords = []
        
            for data in hand_trajectory_data.values():
                timestamps.extend(data['timestamp'])
                x_coords.extend(data['x'])
                y_coords.extend(data['y'])

            # データを時系列でソート
            sorted_indices = np.argsort(timestamps)
            timestamps = np.array(timestamps)[sorted_indices]
            x_coords = np.array(x_coords)[sorted_indices]
            y_coords = np.array(y_coords)[sorted_indices]

            start_time = timestamps[0]
            timestamps = [(timestamp - start_time) for timestamp in timestamps]
            
            # 分散を計算
            x_variances = []
            y_variances = []
            
            for i in range(len(timestamps)):
                start_idx = max(0, i - window_size + 1)
                window_x = x_coords[start_idx:i+1]
                window_y = y_coords[start_idx:i+1]
                
                if len(window_x) > 0:
                    x_variances.append(np.var(window_x))
                    y_variances.append(np.var(window_y))
                else:
                    x_variances.append(0.0)
                    y_variances.append(0.0)

            # グラフの作成
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # X座標の分散プロット
            ax1.plot(timestamps, x_variances, 'b-', label='X Variance', linewidth=2)
            ax1.set_title('X Coordinate Variance', fontsize=14)
            ax1.set_ylabel('Variance', fontsize=12)
            ax1.grid(True)
            ax1.legend(fontsize=10)
            
            # Y座標の分散プロット
            ax2.plot(timestamps, y_variances, 'r-', label='Y Variance', linewidth=2)
            ax2.set_title('Y Coordinate Variance', fontsize=14)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Variance', fontsize=12)
            ax2.grid(True)
            ax2.legend(fontsize=10)

            # レイアウトの調整
            plt.tight_layout()
            
            # 統計情報の表示
            stats_text = (
                f'X Variance - Mean: {np.mean(x_variances):.6f}, Max: {np.max(x_variances):.6f}\n'
                f'Y Variance - Mean: {np.mean(y_variances):.6f}, Max: {np.max(y_variances):.6f}'
            )
            fig.text(0.02, 0.02, stats_text, fontsize=10, transform=fig.transFigure)

            # グラフの保存
            save_path = self.session_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"分散グラフを保存しました: {save_path}")
                

        except Exception as e:
            logger.error(f"分散グラフの作成中にエラー: {e}")
            return None, None, None

    def load_and_create_3d_animation(self, csv_path, output_dir="script"):
        """
        CSVファイルから手の軌跡データを読み込み、3Dアニメーションを作成する
        分散計算を含む
        
        Parameters:
        csv_path (str or Path): 軌跡データのCSVファイルパス
        output_dir (str or Path): アニメーション保存先ディレクトリ
        """
        try:
            # CSVファイルの読み込み
            df = pd.read_csv(csv_path)
            
            # 必要なカラムが存在するか確認
            required_columns = ['timestamp', 'x', 'y', 'z']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSVファイルに必要なカラムがありません。必要なカラム: {required_columns}")
            
            # データを辞書形式に変換
            hand_trajectory_data = {
                'hand': {
                    'timestamp': df['timestamp'].values,
                    'x': df['x'].values,
                    'y': df['y'].values,
                    'z': df['z'].values,
                    'is_palm_up': np.ones(len(df), dtype=bool)  # すべてTrueに設定
                }
            }
            
            # 分散計算用の関数
            def calculate_variance(data, current_frame, window_size=30):
                start_idx = max(0, current_frame - window_size + 1)
                window_data = data[start_idx:current_frame + 1]
                if len(window_data) > 0:
                    return np.var(window_data)
                return 0.0

            def create_3d_trajectory_animation(hand_trajectory_data, output_dir):
                if not hand_trajectory_data:
                    logger.warning("手の軌跡データがありません")
                    return

                # データの取り出し
                data = hand_trajectory_data['hand']
                timestamps = data['timestamp']
                x_coords = data['x']
                y_coords = data['y']
                z_coords = data['z']
                start_tine = timestamps[0]

                # figureとグリッドの設定
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
                ax = fig.add_subplot(gs[0], projection='3d')
                text_ax = fig.add_subplot(gs[1])
                text_ax.axis('off')

                # フォントサイズを設定
                TITLE_SIZE = 16
                LABEL_SIZE = 14
                INFO_SIZE = 12

                # プロットの設定
                ax.set_title('Hand Trajectory (3D)', fontsize=TITLE_SIZE)
                ax.set_xlabel('X', fontsize=LABEL_SIZE)
                ax.set_ylabel('Y', fontsize=LABEL_SIZE)
                ax.set_zlabel('Z', fontsize=LABEL_SIZE)

                # テキスト表示の設定
                time_text = text_ax.text(0.05, 0.90, '', 
                                    transform=text_ax.transAxes, 
                                    fontsize=INFO_SIZE,
                                    fontweight='bold')
                coord_text = text_ax.text(0.05, 0.75, '', 
                                        transform=text_ax.transAxes, 
                                        fontsize=INFO_SIZE,
                                        fontweight='bold')
                
                # 分散表示用のテキスト
                variance_title = text_ax.text(0.05, 0.65, 'Variance (last 30 frames):',
                                            transform=text_ax.transAxes,
                                            fontsize=INFO_SIZE,
                                            fontweight='bold')
                x_var_text = text_ax.text(0.05, 0.60, '',
                                        transform=text_ax.transAxes,
                                        fontsize=INFO_SIZE)
                y_var_text = text_ax.text(0.05, 0.55, '',
                                        transform=text_ax.transAxes,
                                        fontsize=INFO_SIZE)
                z_var_text = text_ax.text(0.05, 0.50, '',
                                        transform=text_ax.transAxes,
                                        fontsize=INFO_SIZE)

                # プロットの初期化
                line = []
                point = ax.plot([], [], [], 'o', c='blue', markersize=8)[0]

                # 軸の範囲設定
                margin = 0.1
                ax.set_xlim([min(x_coords) - margin, max(x_coords) + margin])
                ax.set_ylim([min(y_coords) - margin, max(y_coords) + margin])
                ax.set_zlim([min(z_coords) - margin, max(z_coords) + margin])
                
                ax.grid(True)
                ax.view_init(elev=270, azim=90)

                def update(frame):
                    # 以前のラインをクリア
                    if line:
                        line[0].remove()
                    line.clear()
                    
                    # フレームまでのデータを描画
                    if frame > 0:
                        trajectory_line = ax.plot(x_coords[:frame+1],
                                            y_coords[:frame+1],
                                            z_coords[:frame+1],
                                            c='blue',
                                            alpha=0.7,
                                            linewidth=2)[0]
                        line.append(trajectory_line)
                    
                    # 現在位置の点と情報を更新
                    point.set_data([x_coords[frame]], [y_coords[frame]])
                    point.set_3d_properties([z_coords[frame]])
                    
                    relative_time = timestamps[frame] - start_tine
                    # テキスト情報を更新
                    time_text.set_text(f'Time: {relative_time:.2f} sec')
                    coord_text.set_text(f'Position:\nX: {x_coords[frame]:.2f}\n'
                                    f'Y: {y_coords[frame]:.2f}\n'
                                    f'Z: {z_coords[frame]:.2f}')
                    
                    # 分散を計算して表示
                    x_variance = calculate_variance(x_coords, frame)
                    y_variance = calculate_variance(y_coords, frame)
                    z_variance = calculate_variance(z_coords, frame)
                    x_var_text.set_text(f'X variance: {x_variance:.6f}')
                    y_var_text.set_text(f'Y variance: {y_variance:.6f}')
                    z_var_text.set_text(f'Z variance: {z_variance:.6f}')
                    
                    return line + [point, time_text, coord_text,
                                x_var_text, y_var_text, z_var_text]

                # アニメーションの作成と保存
                num_frames = len(timestamps)
                ani = animation.FuncAnimation(fig, 
                                            update,
                                            frames=num_frames,
                                            interval=50,
                                            blit=True)

                output_path = Path(output_dir) / 'hand_trajectory_3d.mp4'
                writer = animation.FFMpegWriter(fps=20,
                                            metadata=dict(artist='HandTracker'),
                                            bitrate=5000)
                ani.save(str(output_path), writer=writer)
                
                plt.close(fig)
                logger.info(f"3Dアニメーションを保存しました: {output_path}")
                
            # アニメーションの作成
            create_3d_trajectory_animation(hand_trajectory_data, output_dir)
            
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")

    def load_and_plot_trajectory_variance(self, csv_path, window_size=50, save_path="variance_plot.png"):
        """
        手の軌跡データの分散を計算してグラフ化する関数
        
        Parameters:
        -----------
        hand_trajectory_data : dict
            手の軌跡データを含む辞書
        window_size : int
            分散を計算する際のウィンドウサイズ（デフォルト: 30フレーム）
        save_path : str or Path, optional
            グラフを保存するパス。Noneの場合は保存しない
        """
        try:
            output_dir = Path("script")
            # CSVファイルの読み込み
            df = pd.read_csv(csv_path)
            
            # 必要なカラムが存在するか確認
            required_columns = ['timestamp', 'x', 'y', 'z']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSVファイルに必要なカラムがありません。必要なカラム: {required_columns}")
            
            # データを辞書形式に変換
            hand_trajectory_data = {
                'hand': {
                    'timestamp': df['timestamp'].values,
                    'x': df['x'].values,
                    'y': df['y'].values,
                    'z': df['z'].values,
                    'is_palm_up': np.ones(len(df), dtype=bool)  # すべてTrueに設定
                }
            }
            # データの準備
            timestamps = []
            x_coords = []
            y_coords = []
        
            for data in hand_trajectory_data.values():
                timestamps.extend(data['timestamp'])
                x_coords.extend(data['x'])
                y_coords.extend(data['y'])

            # データを時系列でソート
            sorted_indices = np.argsort(timestamps)
            timestamps = np.array(timestamps)[sorted_indices]
            x_coords = np.array(x_coords)[sorted_indices]
            y_coords = np.array(y_coords)[sorted_indices]

            start_time = timestamps[0]
            timestamps = [(timestamp - start_time) for timestamp in timestamps]
            
            # 100秒までのデータに制限
            mask = np.array(timestamps) <= 130
            timestamps = np.array(timestamps)[mask]
            x_coords = x_coords[mask]
            y_coords = y_coords[mask]
            
            # 分散を計算
            x_variances = []
            y_variances = []
            
            for i in range(len(timestamps)):
                start_idx = max(0, i - window_size + 1)
                window_x = x_coords[start_idx:i+1]
                window_y = y_coords[start_idx:i+1]
                
                if len(window_x) > 0:
                    x_variances.append(np.var(window_x))
                    y_variances.append(np.var(window_y))
                else:
                    x_variances.append(0.0)
                    y_variances.append(0.0)

            # グラフの作成
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # X座標の分散プロット
            ax1.plot(timestamps, x_variances, 'b-', label='X Variance', linewidth=2)
            ax1.set_title('X Coordinate Variance', fontsize=14)
            ax1.set_ylabel('Variance', fontsize=12)
            ax1.set_ylim(0, 0.12)
            ax1.grid(True)
            ax1.legend(fontsize=10)
            
            # Y座標の分散プロット
            ax2.plot(timestamps, y_variances, 'r-', label='Y Variance', linewidth=2)
            ax2.set_title('Y Coordinate Variance', fontsize=14)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Variance', fontsize=12)
            ax2.set_ylim(0, 0.12)
            ax2.grid(True)
            ax2.legend(fontsize=10)

            # レイアウトの調整
            plt.tight_layout()
            
            # 統計情報の表示
            stats_text = (
                f'Statistics (window size: {window_size} frames)\n'
                f'X Variance - Mean: {np.mean(x_variances):.6f}, Max: {np.max(x_variances):.6f}\n'
                f'Y Variance - Mean: {np.mean(y_variances):.6f}, Max: {np.max(y_variances):.6f}'
            )
            fig.text(0.02, 0.02, stats_text, fontsize=10, transform=fig.transFigure)

            # グラフの保存
            save_path = output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"分散グラフを保存しました: {save_path}")
                

        except Exception as e:
            logger.error(f"分散グラフの作成中にエラー: {e}")
            return None, None, None

if __name__ == "__main__":
    file_path = "data/20250113_175606Jieさん/hand_trajectories.csv"
    file_path = "data/20250113_180612Jieさん/hand_trajectories.csv"
    
    visualizer = DataVisualizer("script")
    visualizer.load_and_plot_trajectory_variance(file_path)
    # visualizer.load_and_create_3d_animation(file_path)