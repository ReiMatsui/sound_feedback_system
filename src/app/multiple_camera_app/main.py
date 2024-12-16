from datetime import datetime
from pathlib import Path
import cv2
import queue
from loguru import logger
from src.utils.camera_manager import CameraManager
from src.utils.data_recorder import DataRecorder
from src.utils.face_processor import FaceProcessor
from src.utils.hand_processor import HandProcessor
from src.utils.video_recorder import VideoRecorder

class Application:
    """
    手の位置に応じて音を変換して伝える
    顔向きデータ、手の位置データなどを取得
    """
    def __init__(self, face_camera_no: int = 0, hand_camera_no: int = 1):
        
        # 日時ごとのディレクトリ作成
        self.session_dir = self._create_session_dir()
        
        # 手と顔の検出用カメラ初期化
        self.hand_camera_manager = CameraManager(camera_no=hand_camera_no)
        self.face_camera_manager = CameraManager(camera_no=face_camera_no)
        
        # 顔向き、手の座標データcsv作成と可視化
        self.data_recorder = DataRecorder(self.session_dir)
        
        # mediapipeによる手と顔の画像処理
        self.face_processor = FaceProcessor(self.data_recorder)
        self.hand_processor = HandProcessor(self.data_recorder)
        
        # 録画クライアントの初期化
        self.face_video_recorder = VideoRecorder(session_dir=self.session_dir,video_name="face_tracking_video.mp4", width=640, height=360)
        self.hand_video_recorder = VideoRecorder(session_dir=self.session_dir,video_name="hand_tracking_video.mp4", width=640, height=480)
    
    def _create_session_dir(self):
        session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("output") / session_start_time
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"セッションディレクトリを作成しました: {session_dir}")
        return session_dir
    
    def run(self):
        """ 
        メインアプリケーションループ
        """
        cv2.startWindowThread()
        try:
            # 処理スレッドを開始
            self.face_processor.start()
            self.hand_processor.start()
            
            while (self.face_camera_manager.capture.isOpened() and
                   self.hand_camera_manager.capture.isOpened()):
                
                # カメラからフレームを取得
                face_frame = self.face_camera_manager.get_frames()
                hand_frame = self.hand_camera_manager.get_frames()
                if face_frame is None or hand_frame is None:
                    break
                
                # フレームを処理キューに追加
                try:
                    self.face_processor.put_to_queue(face_frame.copy())
                    self.hand_processor.put_to_queue(hand_frame.copy())
                except queue.Full:
                    continue

                # 処理済みの結果を取得
                try:
                    face_results, processed_face_frame = self.face_processor.get_from_queue()
                    hand_results, processed_hand_frame = self.hand_processor.get_from_queue()
                except queue.Empty:
                    continue
                
                face_image = processed_face_frame.copy()
                hand_image = processed_hand_frame.copy()
                
                # 手のランドマーク処理
                if hand_results['multi_hand_landmarks']:
                    self.hand_processor.process_hand_landmarks(hand_image, hand_results)
                
                # 顔のランドマーク処理
                if face_results['multi_face_landmarks']:
                    self.face_processor.process_face_landmarks(face_results)
                    
                # 手と顔のカメラ画面録画    
                self.face_video_recorder.write_frames(face_image)
                self.hand_video_recorder.write_frames(hand_image)
                
                # カメラからの処理済み映像を表示
                self.face_camera_manager.imshow("Face Tracking", face_image)
                self.hand_camera_manager.imshow("Hand Tracking", hand_image)
            
                if cv2.waitKey(1) == ord('q'):
                    break
                
        except Exception as e:
            logger.exception(f"メインループでエラーが発生:{e}")
                    
        finally:
            self.cleanup()
            self.process_data()
    
    def cleanup(self):
        # 別スレッドでのmediapipe処理を終了し、キューをクリア
        self.face_processor.clean_up()
        self.hand_processor.clean_up()
        
        # OpenCVリソースの解放
        for camera_manager in [self.face_camera_manager, self.hand_camera_manager]:
            camera_manager.release()
        for video_recorder in [self.face_video_recorder, self.face_video_recorder]:
            video_recorder.release()

        cv2.destroyAllWindows()
        
    def process_data(self):
        # 各種データをcsvに保存
        self.data_recorder.save_data()
        # データを可視化
        self.data_recorder.visualize_data()
              
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
        
        # アプリケーションの初期化と実行
        app = Application()
        app.run()
        
    except Exception as e:
        logger.exception(f"アプリケーションの起動に失敗{e}")
        
    finally: 
        logger.info("アプリケーションを終了します")

if __name__ == '__main__':
    main()