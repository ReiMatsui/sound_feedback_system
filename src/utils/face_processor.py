import mediapipe as mp
import numpy as np
import cv2
from loguru import logger

class FaceProcessor:
    def __init__(self, ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(image_rgb)
    
    def calculate_orientation(self, landmarks):
        """
        顔の向き（yaw, pitch, roll）を計算
        
        Returns:
            tuple: (yaw, pitch, roll) in degrees
            - yaw: 左右の回転角度 (-: 左, +: 右)
            - pitch: 上下の回転角度 (-: 上, +: 下)
            - roll: 首の傾き (-: 左傾き, +: 右傾き)
        """
        try:
            # 必要なランドマークのインデックス
            # 鼻先
            nose_tip = landmarks.landmark[4]
            # 両目の外側と内側のポイント
            left_eye_outer = landmarks.landmark[33]
            left_eye_inner = landmarks.landmark[133]
            right_eye_inner = landmarks.landmark[362]
            right_eye_outer = landmarks.landmark[263]
            
            # Yawの計算 (左右の回転)
            eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
            eye_distance = abs(left_eye_outer.x - right_eye_outer.x)
            yaw = np.arctan2(nose_tip.x - eye_center_x, eye_distance) * 180 / np.pi
            
            # Pitchの計算 (上下の回転)
            eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
            pitch = np.arctan2(nose_tip.y - eye_center_y, eye_distance) * 180 / np.pi
            
            # Rollの計算 (首の傾き)
            # 両目の傾きから計算
            dy = right_eye_outer.y - left_eye_outer.y
            dx = right_eye_outer.x - left_eye_outer.x
            roll = np.arctan2(dy, dx) * 180 / np.pi
            
            return yaw, pitch, roll
            
        except Exception as e:
            logger.error(f"顔の向き計算中にエラー: {e}")
            return 0, 0, 0