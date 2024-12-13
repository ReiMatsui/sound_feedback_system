import cv2
from pathlib import Path

class VideoRecorder:
    def __init__(self, session_dir: Path, width: int, height: int, fps: float = 20.0):
        self.face_video_writer = self._create_writer(session_dir / 'face_tracking_video.mp4', width, height, fps)
        self.hand_video_writer = self._create_writer(session_dir / 'hand_tracking_video.mp4', width, height, fps)
    
    def _create_writer(self, path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    
    def write_frames(self, face_frame, hand_frame):
        self.face_video_writer.write(face_frame)
        self.hand_video_writer.write(hand_frame)
    
    def release(self):
        self.face_video_writer.release()
        self.hand_video_writer.release()