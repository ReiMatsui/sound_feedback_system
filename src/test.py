import cv2

def main():
    # カメラのIDを指定 (通常、0がデフォルトカメラ、1が2台目のカメラ)
    cam1_id = 0
    cam2_id = 1

    # カメラを初期化
    cap1 = cv2.VideoCapture(cam1_id)
    cap2 = cv2.VideoCapture(cam2_id)

    # カメラが正しくオープンされているか確認
    if not cap1.isOpened():
        print(f"Error: Camera {cam1_id} could not be opened.")
        return
    if not cap2.isOpened():
        print(f"Error: Camera {cam2_id} could not be opened.")
        return

    print("Press 'q' to quit.")

    while True:
        # 各カメラからフレームを取得
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Failed to capture video.")
            break

        # 各カメラの映像をウィンドウに表示
        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
