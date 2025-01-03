from pydantic_settings import BaseSettings
from typing import Literal
from pydantic import Field

class Settings(BaseSettings):
    midi_output_port: int
    face_camera_id: int
    hand_camera_1_id: int
    hand_camera_2_id: int

    class Config:
        env_file = ".env"

setting = Settings()  

if __name__ == "__main__":
    print(setting.midi_output_port)