from draw_hand_landmarks import *
from sound_generator import *
import pygame.midi


def main():
    draw_hand_landmarks_with_sound(sound_gen)
    
if __name__ == "__main__":
    pygame.midi.init()
    try:
        inputID, outputID = SoundGenerator.get_IOdeviceID()
        sound_gen = SoundGenerator(inputID=inputID, outputID=outputID)
        main()        
    except Exception as e:
        logger.exception("An error occurred")  # エラーをログに記録