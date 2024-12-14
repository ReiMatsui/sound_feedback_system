import cv2
from loguru import logger

class CameraManager:
    """
    カメラ映像を取得するクラス
    """
    def __init__(self, camera_no: int = 0, width: int = 640, height: int = 360):
        self.capture = cv2.VideoCapture(camera_no)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frames(self):
        self.ret, self.frame = self.capture.read()
        
        if self.ret:
            return cv2.flip(self.frame, 1)
        return None, None
    
    def imshow(self, window_name: str):
        try:
            cv2.imshow(window_name, self.frame)
        except Exception as e:
            logger.error(f"画像表示/保存中のエラー: {e}")  
            
    def release(self):
        self.capture.release()