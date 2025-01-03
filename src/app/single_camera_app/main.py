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
from src.config import setting

class Application:
    """
    手の位置に応じて音を変換して伝える
    顔向きデータ、手の位置データなどを取得
    """
    def __init__(self, camera_no: int = setting.face_camera_id):
        
        # 日時ごとのディレクトリ作成
        self.session_dir = self._create_session_dir()
        
        # 手と顔の検出用カメラ初期化
        self.camera_manager = CameraManager(camera_no=camera_no)
        
        # 顔向き、手の座標データcsv作成と可視化
        self.data_recorder = DataRecorder(self.session_dir)
        
        # mediapipeによる手と顔の画像処理
        self.face_processor = FaceProcessor(self.data_recorder)
        self.hand_processor = HandProcessor(self.data_recorder)
        
        # 録画クライアントの初期化
        self.video_recorder = VideoRecorder(session_dir=self.session_dir,
                                                 video_name="tracking_video.mp4",
                                                 width=self.camera_manager.width, 
                                                 height=self.camera_manager.height)
    
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
        # self.hand_processor.sound_generator.set_stop_timer(10, 20)
        
        try:
            # 処理スレッドを開始
            self.face_processor.start()
            self.hand_processor.start()
            self.hand_processor.sound_generator.play_rhythm()
            while (self.camera_manager.capture.isOpened()):
                
                # カメラからフレームを取得
                frame = self.camera_manager.get_frames()

                if frame is None:
                    continue
                
                # フレームを処理キューに追加
                try:
                    self.face_processor.put_to_queue(frame.copy())
                except queue.Full:
                    continue

                # 顔の処理済みの結果を取得
                try:
                    face_results, processed_face_frame = self.face_processor.get_from_queue()
                except queue.Empty:
                    continue

                # 顔の処理済みフレームを処理キューに追加
                try:
                    self.hand_processor.put_to_queue(processed_face_frame.copy())
                except queue.Full:
                    continue

                # 処理済みの結果を取得
                try:
                    hand_results, processed_frame = self.hand_processor.get_from_queue()
                except queue.Empty:
                    continue

                processed_image = processed_frame.copy()
                
                # 手のランドマーク処理
                if hand_results['multi_hand_landmarks']:
                    self.hand_processor.process_hand_landmarks(processed_image, hand_results)
                    # self.hand_processor.sound_generator.start_rhythm()
                else:
                    # self.hand_processor.sound_generator.stop_rhythm()
                    self.hand_processor.sound_generator.current_notes = None
                
                # 顔のランドマーク処理
                if face_results['multi_face_landmarks']:
                    self.face_processor.process_face_landmarks(processed_image, face_results)
                    
                # 手と顔のカメラ画面録画    
                self.video_recorder.write_frames(processed_image)
                
                # カメラからの処理済み映像を表示
                self.camera_manager.imshow("Hand Tracking", processed_image)
            
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
        self.camera_manager.release()
        self.video_recorder.release()

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
    logger.info(setting)
    main()