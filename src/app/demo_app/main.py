from src.app.triple_camera_app.main import Application
from loguru import logger
import cv2
import queue
from src.models.point import Point

class DemoApp(Application):
    """デモ用のデータ記録を行わないアプリケーション"""
    def process_data(self):
        pass

    def run(self):
        """ 
        メインアプリケーションループ
        """
        cv2.startWindowThread()
        try:
            # 処理スレッドを開始
            self.face_processor.start()
            self.hand_processor.start()
            self.hand_processor2.start()
            self.hand_processor.sound_generator.play_rhythm()
            self.hand_processor.sound_generator.goal_point = Point(0.5, 0.5, 0.1)

            while (self.face_camera_manager.capture.isOpened() and
                   self.hand_camera_manager.capture.isOpened() and
                   self.hand_camera2_manager.capture.isOpened()):
                
                # カメラからフレームを取得
                face_frame = self.face_camera_manager.get_frames()
                hand_frame = self.hand_camera_manager.get_frames()
                hand_frame2 = self.hand_camera2_manager.get_frames()

                if face_frame is None or hand_frame is None or hand_frame2 is None:
                    break
                
                # フレームを処理キューに追加
                try:
                    self.face_processor.put_to_queue(face_frame.copy())
                    self.hand_processor.put_to_queue(hand_frame.copy())
                    self.hand_processor2.put_to_queue(hand_frame2.copy())
                except queue.Full:
                    continue

                # 処理済みの結果を取得
                try:
                    face_results, processed_face_frame = self.face_processor.get_from_queue()
                    hand_results, processed_hand_frame = self.hand_processor.get_from_queue()
                    hand_results2, processed_hand_frame2 = self.hand_processor2.get_from_queue()
                except queue.Empty:
                    continue
                
                face_image = processed_face_frame.copy()
                hand_image = processed_hand_frame.copy()
                hand_image2 = processed_hand_frame2.copy()
                
                # 手のランドマーク処理
                if hand_results['multi_hand_landmarks']:
                    self.hand_processor.process_hand_landmarks2(hand_image, hand_results, hand_results2)
                else:
                    self.hand_processor.sound_generator.current_notes = None

                # 手の縦方向ランドマーク処理
                # if hand_results2['multi_hand_landmarks']:
                #     self.hand_processor2.process_hand_landmarks(hand_image2, hand_results2)

                # 顔のランドマーク処理
                if face_results['multi_face_landmarks']:
                    self.face_processor.process_face_landmarks(face_image, face_results, self.hand_processor.sound_generator)
                    
                # 手と顔のカメラ画面録画    
                self.face_video_recorder.write_frames(face_image)
                self.hand_video_recorder.write_frames(hand_image)
                self.hand_video_recorder2.write_frames(hand_image2)
                
                # カメラの処理済み映像を表示
                # 実験時はコメントアウト

                # self.face_camera_manager.imshow("Face Tracking", face_image)
                # self.hand_camera_manager.imshow("Hand Tracking", hand_image)
                # self.hand_camera2_manager.imshow("Hand Tracking2", hand_image2)
            
                if cv2.waitKey(1) == ord('q'):
                    break
                
        except Exception as e:
            logger.exception(f"メインループでエラーが発生:{e}")
                    
        finally:
            self.cleanup()
            self.process_data()

def main():
    """
    アプリケーション起動
    """    
    try:
        logger.info("アプリケーションを開始します")
        
        # アプリケーションの初期化と実行
        app = DemoApp()
        app.run()
        
    except Exception as e:
        logger.exception(f"アプリケーションの起動に失敗{e}")
        
    finally: 
        logger.info("アプリケーションを終了します")

if __name__ == '__main__':
    main()