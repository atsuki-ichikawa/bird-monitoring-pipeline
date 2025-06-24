# Windows実行ファイル作成ガイド

このガイドでは、Windows用の実行ファイルを作成する方法を説明します。

## 概要

### 🎯 **推奨アプローチ: 段階的配布**

完全なML依存関係を含む巨大な実行ファイル（～3GB）を作成する代わりに、段階的配布システムを使用します：

1. **軽量ランチャー** (~50MB) - 初期配布
2. **依存関係の自動インストール** - 初回起動時
3. **モデルファイルの自動ダウンロード** - 必要に応じて

## 利点

- ✅ **配布サイズ**: 50MB vs 3GB
- ✅ **メンテナンス**: 簡単な更新とバグ修正
- ✅ **カスタマイズ**: ユーザーが必要な機能のみインストール
- ✅ **エラー対応**: 問題のあるコンポーネントのみ再インストール

## ビルド手順

### 1. 開発環境準備

```bash
# 開発用依存関係インストール
pip install pyinstaller
pip install -r requirements-windows.txt

# Windows固有の依存関係（Windowsマシンで実行）
pip install pywin32
```

### 2. 段階的配布パッケージ作成

```bash
# 自動ビルドスクリプト実行
python build_windows.py

# 出力: dist_windows/BirdMonitoringPipeline_Windows.zip
```

### 3. PyInstaller実行ファイル作成（オプション）

```bash
# 基本的な実行ファイル
pyinstaller --onefile src/cli_minimal.py --name BirdMonitor_CLI

# GUIランチャー付き
pyinstaller --onefile build_windows.py --name BirdMonitor_Launcher --noconsole

# 詳細設定版
pyinstaller bird_monitor.spec
```

## 配布パッケージ構成

```
BirdMonitoringPipeline_Windows.zip
├── install.bat                    # インストーラー
├── launcher.py                    # GUIランチャー
├── src/                          # ソースコード
├── requirements.txt              # 完全な依存関係
├── requirements-minimal.txt      # 基本依存関係
├── README_Windows.txt           # Windows用説明書
└── bird_monitor.spec            # PyInstaller設定
```

## ユーザー向け使用手順

### インストール

1. **ZIPファイルをダウンロード**
2. **適当なフォルダに解凍**
3. **install.bat をダブルクリック**

### 初回起動

1. **ランチャーが自動起動**
2. **「依存関係をインストール」をクリック**
3. **ML依存関係とモデルが自動ダウンロード**
4. **「アプリケーション開始」をクリック**

### 使用方法

- **ブラウザ**: http://localhost:8000 にアクセス
- **CLI**: BirdMonitor_CLI.exe を実行
- **GUI**: launcher.py を実行

## 高度な設定

### カスタムモデル配置

```bash
# 事前にモデルファイルを配置
mkdir models
mkdir models/birdnet
mkdir models/yolo
mkdir models/transfg

# モデルファイルをダウンロード
curl -L "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt" -o models/yolo/yolov10n.pt
```

### GPU対応版

```bash
# GPU用依存関係を追加
echo "tensorflow-gpu>=2.13.0" >> requirements.txt
echo "torch>=2.0.0+cu118" >> requirements.txt
```

### 完全な実行ファイル作成（非推奨）

```bash
# 警告: 非常に大きなファイルになります（3GB+）
pyinstaller --onefile \
  --add-data "models;models" \
  --add-data "src/web/templates;templates" \
  --add-data "src/web/static;static" \
  --hidden-import="tensorflow" \
  --hidden-import="cv2" \
  --hidden-import="torch" \
  src/web/app.py
```

## トラブルシューティング

### 依存関係エラー

```bash
# 仮想環境をクリーンに再作成
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements-minimal.txt
```

### モデルダウンロードエラー

```bash
# 手動でモデルをダウンロード
mkdir models\yolo
curl -L "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt" -o models\yolo\yolov10n.pt
```

### メモリ不足エラー

```bash
# 軽量版を使用
python -m src.cli_minimal status
python -m src.web.app_minimal
```

## パフォーマンス最適化

### 1. 軽量モデル使用

```python
# config.yamlで軽量モデルを指定
yolo_model_path: "models/yolo/yolov10n.pt"  # nano版
video_frame_skip: 3  # 3フレームごとに処理
batch_size: 8  # バッチサイズを削減
```

### 2. GPU加速

```python
# GPU使用設定
gpu_enabled: true
device: "cuda"  # またはauto
```

### 3. メモリ管理

```python
# メモリ使用量制限
max_memory_usage: 4096  # MB
cleanup_temp_files: true
```

## セキュリティ考慮事項

### 1. 実行ファイル署名

```bash
# コード署名（証明書が必要）
signtool sign /f certificate.pfx /p password BirdMonitor.exe
```

### 2. ウイルス対策ソフト対応

```bash
# 誤検知を避けるため
# - 信頼できるソースからのみ配布
# - VirusTotalでスキャン
# - READMEに説明を記載
```

## まとめ

段階的配布アプローチにより、Windows用の配布を効率的に行えます：

- **初期配布**: 50MB程度の軽量パッケージ
- **自動セットアップ**: ユーザーフレンドリーなインストール
- **柔軟性**: 必要な機能のみインストール
- **保守性**: 簡単な更新とバグ修正

この方法により、重いML依存関係を含む複雑なアプリケーションでも、ユーザーフレンドリーな配布が可能になります。