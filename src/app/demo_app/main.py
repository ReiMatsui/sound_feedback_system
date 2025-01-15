from src.app.triple_camera_app.main import Application
from loguru import logger

class DemoApp(Application):
    def process_data(self):
        pass

def main():
    """
    アプリケーション起動
    """    
    try:
        # ログの設定
        logger.add(
            "logs/app_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            encoding="utf-8"
        )
        logger.info("アプリケーションを開始します")
        
        # アプリケーションの初期化と実行
        app = DemoApp()
        app.run()
        
    except Exception as e:
        logger.exception(f"アプリケーションの起動に失敗{e}")
        
    finally: 
        logger.info("アプリケーションを終了します")

if __name__ == '__main__':
    main()