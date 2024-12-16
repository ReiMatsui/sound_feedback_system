import cv2
import mediapipe as mp
import threading
import queue
import numpy as np
from loguru import logger
from src.utils.sound_generator import SoundGenerator
from src.utils.data_recorder import DataRecorder

class HandProcessor:
    """ 
    mediapipeで手の処理を行うクラス
    """
    def __init__(self, data_recorder: DataRecorder):
        # 音ジェネレーター設定
        try:
            output_names = SoundGenerator.get_output_names()
            self.sound_generator = SoundGenerator(output_name=output_names[0])
        except Exception as e:
            logger.exception(f"音ジェネレーターの初期化に失敗:{e}")
            raise
        
        self.data_recorder = data_recorder
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hand_frame_queue = queue.Queue(maxsize=10)
        self.hand_result_queue = queue.Queue(maxsize=10)
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
        for q in [self.hand_frame_queue, self.hand_result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.process_thread.join(timeout=2.0)
        self.sound_generator.end()
                
    def put_to_queue(self, frame):
        self.hand_frame_queue.put(frame, timeout=0.1)

    def get_from_queue(self):
        hand_results, processed_hand_frame  = self.hand_result_queue.get(timeout=0.1)
        return hand_results, processed_hand_frame 
    
    def judge_palm_up(self, landmarks, handedness) -> bool:
        """
        手のひらの向きを計算
        
        手のひらの法線ベクトルを計算し、上向きかどうかを判定
        landmark[0]: 手首
        landmark[5, 9, 13, 17]: 指の付け根
        """
        try:
            if_left = (handedness == "Left")
            # 人差し指の付け根
            point5 = np.array([landmarks.landmark[5].x,
                                  landmarks.landmark[5].y])
            # 小指の付け根
            point17 = np.array([landmarks.landmark[17].x,
                                  landmarks.landmark[17].y])          
            
            vector = point17 - point5
            
            if_up = (landmarks.landmark[8].x < landmarks.landmark[20].x) if if_left else (landmarks.landmark[8].x > landmarks.landmark[20].x )
            
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            if_horizontal = (angle < 30) or (angle > 150)
        
            is_palm_up = if_horizontal and if_up # 閾値は調整可能
            
            return is_palm_up
            
        except Exception as e:
            logger.error(f"手のひらの向き計算中にエラー: {e}")
            return False
        
    def process_frame(self):
        """
        別スレッドで手のMediaPipe処理を実行
        """
        try:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            ) as hands:
                while self.running.is_set():
                    try:
                        frame = self.hand_frame_queue.get(timeout=1.0)
                        if frame is None:
                            continue
                        
                        frame_copy = frame.copy()
                        image_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                        hands_results = hands.process(image_rgb)
                        
                        results = {
                            'multi_hand_landmarks': [
                                landmark.copy() if hasattr(landmark, 'copy') 
                                else landmark 
                                for landmark in (hands_results.multi_hand_landmarks or [])
                            ],
                            'handedness': hands_results.multi_handedness
                        }
                        
                        self.hand_result_queue.put((results, frame_copy))
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.exception(f"{e}:手フレーム処理中にエラーが発生")
                        continue
                        
        except Exception as e:
            logger.exception(f"{e}:手のMediaPipe処理スレッドでエラーが発生")
        finally:
            logger.info("手のMediaPipe処理スレッドを終了します")

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
    def process_hand_landmarks(self, image, hand_results):
        """
        手のランドマークの処理と描画を行う
        """
        try:
            for i, landmarks in enumerate(hand_results['multi_hand_landmarks']):
                # ランドマークの描画
                self.draw_landmarks(image, landmarks)
                
                # 手の位置データ保存
                self.data_recorder.record_hand_trajectory(landmarks, i)
                
                # サウンドジェネレーターの更新（最初の手のみ）
                if i == 0 and self.sound_generator is not None:
                    hand_x = landmarks.landmark[9].x
                    hand_y = landmarks.landmark[9].y
                    handedness = hand_results['handedness'][0].classification[0].label
                    
                    # 手のひらが上向きか判定
                    is_palm_up = self.judge_palm_up(landmarks, handedness)
  
                    new_notes = self.sound_generator.new_notes(hand_x, hand_y, is_palm_up)
                    self.sound_generator.update_notes(new_notes)
                    
                    # Palm upの状態を表示
                    cv2.putText(image, f'Palm up: {is_palm_up}', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        except Exception as e:
            logger.error(f"ハンドランドマーク処理中のエラー: {e}")