import cv2
from loguru import logger

class CameraManager:
    def __init__(self, face_camera_no: int = 0, hand_camera_no: int = 1, width: int = 640, height: int = 360):
        self.face_capture = cv2.VideoCapture(face_camera_no)
        self.hand_capture = cv2.VideoCapture(hand_camera_no)
        
        for capture in [self.face_capture, self.hand_capture]:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get_frames(self):
        face_ret, face_frame = self.face_capture.read()
        hand_ret, hand_frame = self.hand_capture.read()
        
        if face_ret and hand_ret:
            return cv2.flip(face_frame, 1), cv2.flip(hand_frame, 1)
        return None, None
    
    def release(self):
        self.face_capture.release()
        self.hand_capture.release()