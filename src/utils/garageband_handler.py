import subprocess
import signal
import sys
import time
import os

class GarageBandHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.process = None
        
    def open_file(self):
        try:
            # ファイルが存在するか確認
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            # GarageBandでファイルを開く
            self.process = subprocess.Popen(['open', '-a', 'GarageBand', self.file_path])
            print(f"Opened GarageBand with file: {self.file_path}")
            time.sleep(5)
            
        except Exception as e:
            print(f"Error opening file: {e}")
            sys.exit(1)
    
    def close_file(self):
        try:
            if self.process:
                # GarageBandプロセスを終了
                subprocess.run(['osascript', '-e', 'tell application "GarageBand" to quit'])
                print("Closed GarageBand")
        except Exception as e:
            print(f"Error closing GarageBand: {e}")

def signal_handler(sig, frame):
    if handler:
        handler.close_file()
    sys.exit(0)

# シグナルハンドラを設定
signal.signal(signal.SIGINT, signal_handler)

# 使用例
if __name__ == "__main__":
    file_path = "/Users/matsuirei/Music/GarageBand/sample.band"
    handler = GarageBandHandler(file_path)
    
    # ファイルを開く
    handler.open_file()
    
    try:
        # メインプログラムの処理をここに書く
        print("Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # プログラム終了時の処理
        handler.close_file()