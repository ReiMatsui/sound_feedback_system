from datetime import datetime
from pathlib import Path
import cv2
from loguru import logger
from src.sound_generator import SoundGenerator
from src.utils.camera_manager import CameraManager
from src.utils.data_recorder import DataRecorder
from src.utils.data_visualizer import DataVisualizer
from src.utils.face_processor import FaceProcessor
from src.utils.hand_processor import HandProcessor
from src.utils.video_recorder import VideoRecorder

class DualCameraHandFaceSoundTracker:
    def __init__(self, face_camera_no: int = 0, hand_camera_no: int = 1):
        self.session_dir = self._create_session_dir()
        
        self.camera_manager = CameraManager(face_camera_no, hand_camera_no)
        self.face_processor = FaceProcessor()
        self.hand_processor = HandProcessor()
        self.data_recorder = DataRecorder(self.session_dir)
        self.video_recorder = VideoRecorder(self.session_dir, 640, 360)
        self.data_visualizer = DataVisualizer(self.session_dir)
        output_names = SoundGenerator.get_output_names()
        self.sound_generator = SoundGenerator(output_name=output_names[0])
        
        self.running = True
    
    def _create_session_dir(self):
        session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("output") / session_start_time
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def run(self):
        try:
            while self.running:
                face_frame, hand_frame = self.camera_manager.get_frames()
                if face_frame is None or hand_frame is None:
                    break
                
                face_results = self.face_processor.process_frame(face_frame)
                hand_results = self.hand_processor.process_frame(hand_frame)
                
                self._process_face_results(face_results, face_frame)
                self._process_hand_results(hand_results, hand_frame)
                
                self.video_recorder.write_frames(face_frame, hand_frame)
                
                if cv2.waitKey(3) == ord('q'):
                    break
                    
        finally:
            self._cleanup()
    
    def _process_face_results(self, results, frame):
        if results.multi_face_landmarks:
            orientation = self.face_processor.calculate_orientation(results.multi_face_landmarks[0])
            self.data_recorder.record_face_orientation(*orientation)
    
    def _process_hand_results(self, results, frame):
        if results.multi_hand_landmarks:
            for i, landmarks in enumerate(results.multi_hand_landmarks):
                self.hand_processor.draw_landmarks(frame, landmarks)
                self.data_recorder.record_hand_trajectory(landmarks, i)
                
                if i == 0:
                    self._update_sound(landmarks, results.multi_handedness[0])
    
    def _update_sound(self, landmarks, handedness):
        self.sound_generator.update_hand_orientation(landmarks, handedness)
        hand_x = landmarks.landmark[9].x
        hand_y = landmarks.landmark[9].y
        new_notes = self.sound_generator.new_notes(hand_x, hand_y)
        self.sound_generator.update_notes(new_notes)
    
    def _cleanup(self):
        self.camera_manager.release()
        self.video_recorder.release()
        self.data_recorder.save_data()
        self.data_visualizer.create_face_orientation_plots(self.data_recorder.face_orientation_data)
        self.data_visualizer.create_3d_trajectory_animation(self.data_recorder.hand_trajectory_data)
        self.sound_generator.end()
        cv2.destroyAllWindows()
        
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
        
        # トラッカーの初期化と実行
        tracker = DualCameraHandFaceSoundTracker()
        tracker.run()
        
    except Exception as e:
        logger.exception("アプリケーションの起動に失敗")
    
    finally:
        # 最終的なクリーンアップ
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # ウィンドウを確実に閉じる

if __name__ == '__main__':
    main()