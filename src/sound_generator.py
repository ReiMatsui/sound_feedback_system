import pygame.midi
import time
import threading
import math
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# ログの基本設定
logger.add("app.log", rotation="1 MB", level="INFO", format="{time} {level} {message}")

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

class SoundGenerator:
    def __init__(self,inputID,outputID):
        self.inputID=inputID
        self.outputID=outputID
        self.player=pygame.midi.Output(self.outputID)
        self.volume = 50
        self.major_notes = [60, 64, 67, 72, 76, 79] #C_major
        self.minor_notes = [60, 64, 69, 72, 76, 81] #a_minor
        self.dissonance_notes = [60, 58, 57, 56, 55, 54]
        self.current_notes = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.goal_point ={"x": 0.5, "y": 0.5}
    
    def update_notes(self, new_notes):
        with self.lock:
            if self.current_notes == new_notes:
                return
            if self.current_notes is not None:
                try:
                    self.stop_notes(self.current_notes)
                except Exception as e:
                    print(f"Error stopping notes: {e}")

            self.current_notes = new_notes

            # 新しいノートを再生する
            try:
                self.play_notes(new_notes)
            except Exception as e:
                print(f"Error playing notes: {e}")
      
    def play_notes(self, notes):
        for i in range(len(notes)):
            self.player.note_on(notes[i], self.volume, channel=i)
            
    def stop_notes(self, notes):
        for i in range(len(notes)):
            self.player.note_off(notes[i], self.volume, channel=i)
   
    def new_notes(self, x, y):
        dist = calculate_distance([x, y], [self.goal_point["x"], self.goal_point["y"]])
        if dist < 0.1:
            return [60, 64, 67, 72, 76, 79]
        else:
            return [60 - math.floor(dist*10)] 
    
    def end(self):
        self.executor.shutdown(wait=True)
        self.player.close()
        del self.player
        pygame.midi.quit()



def get_IOdeviceID():  
    if pygame.midi.get_count() == 0:
        raise ValueError("No device found")
    for i in range(pygame.midi.get_count()):
        interf, name, inputID, outputID, opened = pygame.midi.get_device_info(i)
        if outputID:  # 出力デバイスのみ表示
            logger.info(f"入力IDは{inputID},出力IDは{outputID}です") 
    return inputID, outputID

if __name__ == "__main__":
    pygame.init()
    pygame.midi.init()
    try:
        inputID, outputID = get_IOdeviceID()
        sound_gen = SoundGenerator(inputID=inputID, outputID=outputID)
        
        sound_gen.play_major_notes()
        time.sleep(3)
        sound_gen.stop_major_notes()
        
        time.sleep(1)
        sound_gen.play_dissonance_notes()
        time.sleep(3)
        sound_gen.stop_dissonance_notes()
        
    except Exception as e:
        logger.exception("An error occurred")  # エラーをログに記録
    finally:
        sound_gen.end()
        
    