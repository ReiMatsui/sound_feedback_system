# sound_feedback_system
## 環境構築
* Python 3.10.15
* Poetry 1.8.3(以下でダウンロード)
```
curl -sSL https://install.python-poetry.org | python3 -
```
## セットアップ
```
git clone https://github.com/ReiMatsui/nakamura_lab_research
cd matsui_research
poetry install
```
## 起動方法
### Ubuntuの場合
* timidityを起動しておく
```
timidity -iA
```
### Macの場合
* GarageBandでプロジェクトを起動しておく
### 実行
* 顔と手を3つのカメラで検出する場合 
``` 
poetry run python src/app/triple_camera_app/main.py
```

* 顔と手を2つのカメラで検出する場合 
``` 
poetry run python src/app/double_camera_app/main.py
```

* 一つのカメラで検出する場合
``` 
poetry run python src/app/single_camera_app/main.py
```

