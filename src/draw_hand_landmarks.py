import cv2
import mediapipe as mp
import time
from sound_generator import *
from loguru import logger
import pygame.midi
import os
os.environ['no_proxy'] = "*"


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# カメラキャプチャの設定
camera_no = 0
video_capture = cv2.VideoCapture(camera_no)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


# 音を再生する関数をスレッド内で実行
def play_sound_conditionally(SoundGenerator: SoundGenerator, landmark):
    if landmark.y < 0.5:
        SoundGenerator.update_notes(SoundGenerator.dissonance_notes)
    else:
        SoundGenerator.update_notes(SoundGenerator.major_notes)

def draw_hand_landmarks_with_sound(SoundGenerator: SoundGenerator):
    with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
        try:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    logger.info("カメラの取得できず")
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                hands_results = hands.process(image)

                image.flags.writeable = True
                write_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 画像の中央の座標を計算
                height, width, _ = write_image.shape
                center_coordinates = (width // 2, height // 2)  # 中央の座標

                # 点の半径と色を指定
                radius = 5
                color = (0, 255, 0)  # 緑色 (BGR)

                # 中央に点を描画
                cv2.circle(write_image, center_coordinates, radius, color, -1)  # -1は円を塗りつぶす


                if hands_results.multi_hand_landmarks:
                    for landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            write_image,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                        new_notes = SoundGenerator.new_notes(landmarks.landmark[9].x, landmarks.landmark[9].y)
                        SoundGenerator.update_notes(new_notes)
                        logger.info(landmarks.landmark[9])

                cv2.imshow('hands', write_image)
                key = cv2.waitKey(10)
                if key == 27:
                    SoundGenerator.end()
                    cv2.waitKey(10)
                    video_capture.release()
                    cv2.waitKey(10)
                    cv2.destroyWindow("hands")
                    cv2.waitKey(10)
                    break
                time.sleep(0.01)
                
        finally:
            logger.info("カメラとウィンドウリソースを解放しました")


            
def draw_hand_landmarks():
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:        
        try:
            while video_capture.isOpened():
                # カメラ画像の取得
                ret, frame = video_capture.read()
                if ret is False:
                    print("カメラの取得できず")
                    break

                # 鏡になるよう反転
                frame = cv2.flip(frame, 1)

                # OpenCVとMediaPipeでRGBの並びが違うため、処理前に変換しておく。
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # 推論処理
                hands_results = hands.process(image)
                # 前処理の変換を戻しておく。
                image.flags.writeable = True
                write_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 有効なランドマーク（今回で言えば手）が検出された場合、ランドマークを描画します。
                if hands_results.multi_hand_landmarks:
                    for landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            write_image,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                cv2.startWindowThread()
                # ディスプレイ表示
                cv2.imshow('hands', write_image)
                key = cv2.waitKey(10)
                if key == 27:  # ESCが押されたら終了
                    print("終了")
                    break


        finally:
            video_capture.release()
            time.sleep(1)
            cv2.destroyWindow("hands")
            
def main():
    draw_hand_landmarks()
    
if __name__ == '__main__':
    pygame.init()
    pygame.midi.init()
    try:
        inputID, outputID = SoundGenerator.get_IOdeviceID()
        sound_gen = SoundGenerator(inputID=inputID, outputID=outputID)
        
        main()
        
    except Exception as e:
        logger.exception("An error occurred")  # エラーをログに記録