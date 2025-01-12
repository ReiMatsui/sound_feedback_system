import mediapipe as mp
import numpy as np
import cv2
import threading 
import queue
from loguru import logger
from src.utils.data_recorder import DataRecorder
from src.utils.sound_generator import SoundGenerator

class FaceProcessor:
    """ 
    mediapipeで顔の処理を行うクラス
    """
    def __init__(self, data_recorder: DataRecorder):
        self.data_recorder = data_recorder
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_frame_queue = queue.Queue(maxsize=10)
        self.face_result_queue = queue.Queue(maxsize=10)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.running = threading.Event()
        self.process_thread = threading.Thread(target=self.process_frame)
        self.process_thread.daemon = True
        self.running.set()
    
    def start(self):
        self.process_thread.start()
        return
    
    def clean_up(self):
        """
        別スレッドでのmediapipe処理を終了し、キューをクリア
        """
        self.running.clear()
        # キューをクリア
        for q in [self.face_frame_queue, self.face_result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.process_thread.join(timeout=2.0)
    
    def put_to_queue(self, frame):
        self.face_frame_queue.put(frame, timeout=0.1)

    def get_from_queue(self):
        face_results, processed_face_frame  = self.face_result_queue.get(timeout=0.1)
        return face_results, processed_face_frame 

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            None,
            self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
    def process_frame(self):
        """
        別スレッドで顔のMediaPipe処理を実行
        """
        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                while self.running.is_set():
                    try:
                        frame = self.face_frame_queue.get(timeout=1.0)
                        if frame is None:
                            continue
                        
                        frame_copy = frame.copy()
                        image_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                        face_results = face_mesh.process(image_rgb)
                        
                        results = {
                            'multi_face_landmarks': [
                                landmark.copy() if hasattr(landmark, 'copy') 
                                else landmark 
                                for landmark in (face_results.multi_face_landmarks or [])
                            ] if face_results.multi_face_landmarks else None
                        }
                        
                        self.face_result_queue.put((results, frame_copy))
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.exception(f"{e}:顔フレーム処理中にエラーが発生")
                        continue
                        
        except Exception as e:
            logger.exception(f"{e}:MediaPipe処理スレッドでエラーが発生")
        finally:
            logger.info("顔のMediaPipe処理スレッドを終了します")
    
    def process_face_landmarks(self, face_image, face_results, sound_generator: SoundGenerator=None):
        """
        顔の向き処理
        """
        try:
            face_landmarks = face_results['multi_face_landmarks'][0]
            # self.draw_landmarks(face_image, face_landmarks)
            yaw, pitch, roll = self.calculate_face_orientation(face_landmarks)
            
            self.data_recorder.record_face_orientation(yaw, pitch, roll)
            
            if len(self.data_recorder.face_orientation_data) > 1:
                diff = abs(self.data_recorder.face_orientation_data[-2][-3] - self.data_recorder.face_orientation_data[-1][-3])
                logger.info(diff)
            
            if sound_generator:
                cv2.putText(face_image, f'sound_on: {sound_generator.is_active}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(face_image, f'yaw: {yaw:.2f}', 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(face_image, f'pitch: {pitch:.2f}', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(face_image, f'roll: {roll:.2f}', 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except Exception as e:
            logger.error(f"顔の向き処理中のエラー: {e}")
            
    def calculate_face_orientation(self, landmarks):
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
            
            # left_eye_inner = landmarks.landmark[133]
            # right_eye_inner = landmarks.landmark[362]
            
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