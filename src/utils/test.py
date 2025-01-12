from vpython import *
import pandas as pd
import numpy as np
import time

class DetailedFaceVisualizer:
    def __init__(self):
        """より詳細な3D顔向きビジュアライザーの初期化"""
        # ウィンドウとシーンの設定
        scene.width = 800
        scene.height = 600
        scene.background = color.white
        scene.camera.pos = vector(0, 0, 15)  # カメラ位置の調整
        
        # 頭部の作成（より楕円に近い形状）
        self.head = ellipsoid(
            pos=vector(0, 0, 0),
            length=2.2,  # 前後方向
            height=2.8,  # 上下方向
            width=2.0,   # 左右方向
            color=color.hsv_to_rgb(vector(0.1, 0.3, 0.9)),  # 肌色
            opacity=1
        )
        
        # 顔のパーツの作成
        # 目（左）
        self.left_eye = compound([
            # 白目
            ellipsoid(
                pos=vector(-0.5, 0.3, 0.9),
                length=0.4,
                height=0.3,
                width=0.1,
                color=color.white
            ),
            # 黒目
            sphere(
                pos=vector(-0.5, 0.3, 1.0),
                radius=0.15,
                color=color.black
            )
        ])
        
        # 目（右）
        self.right_eye = compound([
            # 白目
            ellipsoid(
                pos=vector(0.5, 0.3, 0.9),
                length=0.4,
                height=0.3,
                width=0.1,
                color=color.white
            ),
            # 黒目
            sphere(
                pos=vector(0.5, 0.3, 1.0),
                radius=0.15,
                color=color.black
            )
        ])
        
        # 眉毛（左）
        self.left_eyebrow = cylinder(
            pos=vector(-0.7, 0.7, 0.9),
            axis=vector(0.4, 0.1, 0),
            radius=0.05,
            color=color.black
        )
        
        # 眉毛（右）
        self.right_eyebrow = cylinder(
            pos=vector(0.3, 0.7, 0.9),
            axis=vector(0.4, 0.1, 0),
            radius=0.05,
            color=color.black
        )
        
        # 鼻
        self.nose = compound([
            # 鼻筋
            box(
                pos=vector(0, 0, 1.1),
                length=0.2,
                height=0.6,
                width=0.1,
                color=color.hsv_to_rgb(vector(0.1, 0.4, 0.85))
            ),
            # 鼻先
            sphere(
                pos=vector(0, -0.2, 1.15),
                radius=0.2,
                color=color.hsv_to_rgb(vector(0.1, 0.4, 0.85))
            )
        ])
        
        # 口
        self.mouth = ellipsoid(
            pos=vector(0, -0.8, 0.9),
            length=0.8,
            height=0.2,
            width=0.1,
            color=color.red
        )
        
        # 方向を示す補助線（後頭部から前面への矢印）
        self.direction_indicator = arrow(
            pos=vector(0, 0, -1.1),
            axis=vector(0, 0, 3),
            shaftwidth=0.1,
            color=color.blue,
            opacity=0.3
        )
        
        # テキスト表示用のラベル
        self.info_label = label(
            pos=vector(0, 3, 0),
            text='Yaw: 0° Pitch: 0° Roll: 0°',
            height=16,
            box=False,
            color=color.black
        )
        
        # すべてのパーツをグループ化
        self.face_parts = [self.left_eye, self.right_eye, 
                          self.left_eyebrow, self.right_eyebrow,
                          self.nose, self.mouth, self.direction_indicator]

    def update_orientation(self, yaw, pitch, roll):
        """
        顔の向きを更新
        :param yaw: 左右の回転角（度）
        :param pitch: 上下の回転角（度）
        :param roll: 傾きの角度（度）
        """
        # 角度をラジアンに変換
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # 回転行列の作成
        # Yaw rotation (around Y axis)
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        # Pitch rotation (around X axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Roll rotation (around Z axis)
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # 合成回転行列
        R = Rz @ Rx @ Ry
        
        # 頭部と顔のパーツの向きを更新
        self.head.up = vector(R[1][0], R[1][1], R[1][2])
        self.head.axis = vector(R[2][0], R[2][1], R[2][2])
        
        # すべての顔のパーツに同じ回転を適用
        for part in self.face_parts:
            part.up = vector(R[1][0], R[1][1], R[1][2])
            part.axis = vector(R[2][0], R[2][1], R[2][2])
        
        # 情報表示を更新
        self.info_label.text = f'Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Roll: {roll:.1f}°'

def visualize_face_orientation_data(data_df):
    """
    データフレームから顔の向きデータを可視化
    :param data_df: timestamp, yaw, pitch, rollを含むデータフレーム
    """
    vis = DetailedFaceVisualizer()
    
    for _, row in data_df.iterrows():
        # データを更新
        vis.update_orientation(row['yaw'], row['pitch'], row['roll'])
        # 更新頻度を制御（必要に応じて調整）
        rate(30)

# 使用例
if __name__ == "__main__":
    # サンプルデータの作成
    data = {
        'timestamp': np.arange(0, 10, 0.1),
        'yaw': 30 * np.sin(np.arange(0, 10, 0.1)),
        'pitch': 20 * np.cos(np.arange(0, 10, 0.1)),
        'roll': 10 * np.sin(2 * np.arange(0, 10, 0.1))
    }
    df = pd.DataFrame(data)
    
    # 可視化の実行
    visualize_face_orientation_data(df)