# macOS セットアップガイド

Bird Monitoring Pipeline をmacOS環境で動作させるための包括的なガイドです。

## 🍎 対応環境

- **macOS**: 10.15 Catalina 以降
- **アーキテクチャ**: Apple Silicon (M1/M2/M3) & Intel
- **Python**: 3.11-3.12 (推奨), 3.13 (制限あり)
- **RAM**: 8GB以上 (16GB推奨)

## ⚡ クイックスタート

### 1. 環境テスト
```bash
# リポジトリをクローン
git clone https://github.com/atsuki-ichikawa/bird-monitoring-pipeline.git
cd bird-monitoring-pipeline

# 現在の環境をテスト
python test_macos.py
```

### 2. 軽量版で開始（推奨）
```bash
# 基本依存関係をインストール
pip install -r requirements-minimal.txt
pip install pydantic-settings

# 軽量版を起動
python -m src.web.app_minimal

# ブラウザで http://localhost:8000 にアクセス
```

## 🔧 完全セットアップ

### Python 3.13での注意事項

**⚠️ 重要**: TensorFlowはPython 3.13を完全サポートしていません。完全な機能を使用するには：

#### Option A: pyenv で Python バージョン管理
```bash
# pyenv インストール
brew install pyenv

# Python 3.12 をインストール
pyenv install 3.12.7
pyenv local 3.12.7

# 仮想環境作成
python -m venv venv_312
source venv_312/bin/activate
```

#### Option B: Python 3.13 で可能な範囲で使用
```bash
# 軽量版のみ利用（ML機能は制限）
pip install -r requirements-minimal.txt
python -m src.web.app_minimal
```

### システム依存関係のインストール

#### Homebrew経由（推奨）
```bash
# FFmpeg（動画処理）
brew install ffmpeg

# 音声処理ライブラリ
brew install portaudio libsndfile pkg-config

# 開発ツール
brew install git python@3.12
```

#### 手動インストール
```bash
# Xcode Command Line Tools
xcode-select --install

# Homebrew インストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ML依存関係のインストール

#### Apple Silicon Mac (M1/M2/M3)
```bash
# TensorFlow (Apple最適化版)
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.1

# PyTorch (Apple Silicon対応)
pip install torch torchvision torchaudio

# コンピュータビジョン
pip install opencv-python==4.8.1.78

# 音声処理
pip install librosa==0.10.1 soundfile

# オブジェクト検出
pip install ultralytics==8.0.196

# その他のML依存関係
pip install scikit-learn numpy scipy pandas matplotlib
```

#### Intel Mac
```bash
# 標準 TensorFlow
pip install tensorflow==2.13.0

# その他は Apple Silicon と同じ
pip install torch torchvision torchaudio
pip install opencv-python librosa ultralytics
pip install scikit-learn numpy scipy pandas matplotlib
```

## 🚀 自動セットアップ

### ワンコマンドセットアップ
```bash
# 自動セットアップスクリプトを実行
python setup_macos.py

# このスクリプトは以下を自動実行：
# 1. システム要件チェック
# 2. Homebrew依存関係インストール  
# 3. 仮想環境作成
# 4. アーキテクチャに応じたML依存関係インストール
# 5. 起動スクリプト作成
```

### セットアップ完了後
```bash
# 仮想環境をアクティベート
source venv/bin/activate

# アプリケーション起動
./start_macos.sh
```

## 🧪 動作確認

### 基本機能テスト
```bash
# CLI テスト
python -m src.cli_minimal status

# Web インターフェーステスト
python -m src.web.app_minimal

# 単体テスト実行
pytest tests/test_minimal.py -v
```

### ML機能テスト
```bash
# TensorFlow動作確認
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} - GPU: {tf.config.list_physical_devices(\"GPU\")}')"

# OpenCV動作確認  
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# PyTorch動作確認
python -c "import torch; print(f'PyTorch {torch.__version__} - MPS: {torch.backends.mps.is_available()}')"
```

## 🎯 起動オプション

### 軽量版（テスト用）
```bash
# Web インターフェース
python -m src.web.app_minimal

# CLI
python -m src.cli_minimal status
python -m src.cli_minimal process test_video.mp4
```

### 完全版
```bash  
# Web インターフェース
python -m src.web.app

# CLI  
python -m src.cli process video.mp4 --output results/
```

### 起動スクリプト使用
```bash
# macOS用起動スクリプト（自動作成される）
./start_macos.sh

# 手動作成する場合
cat > start_macos.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python -m src.web.app_minimal &
echo "Bird Monitoring Pipeline started"
echo "Access: http://localhost:8000"
wait
EOF
chmod +x start_macos.sh
```

## 🛠️ トラブルシューティング

### よくある問題と解決方法

#### 1. TensorFlow インストールエラー
```bash
# Python バージョン確認
python --version

# Python 3.13の場合
echo "TensorFlowはPython 3.13を完全サポートしていません"
echo "Python 3.12の使用を推奨します"

# 代替案: 軽量版を使用
pip install -r requirements-minimal.txt
```

#### 2. OpenCV インストールエラー
```bash
# システム依存関係を確認
brew install cmake pkg-config

# 再インストール
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

#### 3. FFmpeg not found エラー
```bash
# Homebrewでインストール
brew install ffmpeg

# パス確認
which ffmpeg
export PATH="/opt/homebrew/bin:$PATH"  # Apple Silicon
export PATH="/usr/local/bin:$PATH"     # Intel
```

#### 4. Permission denied エラー
```bash
# スクリプトに実行権限を付与
chmod +x setup_macos.py
chmod +x start_macos.sh

# 仮想環境の権限修正
chmod -R 755 venv/
```

#### 5. メモリ不足エラー
```bash
# バッチサイズを削減
echo "batch_size: 8" >> config.yaml

# フレームスキップを増加
echo "video_frame_skip: 3" >> config.yaml
```

### パフォーマンス最適化

#### Apple Silicon最適化
```bash
# MPS (Metal Performance Shaders) 有効化
export PYTORCH_ENABLE_MPS_FALLBACK=1

# TensorFlow Metal 最適化
export TF_METAL_DEVICE_PLACEMENT=1
```

#### メモリ使用量削減
```python
# config.yaml
max_memory_usage: 4096  # MB
cleanup_temp_files: true
use_mixed_precision: true
```

#### GPU加速設定
```python
# Apple Silicon GPU使用
device: "mps"  # PyTorch
# または
device: "/GPU:0"  # TensorFlow Metal
```

## 📊 パフォーマンス目安

### Apple Silicon (M1/M2/M3)
- **軽量版**: 即座に起動
- **完全版**: 初回モデルロード 30-60秒
- **動画処理**: ~2-4x リアルタイム（720p）
- **メモリ使用量**: 2-8GB

### Intel Mac
- **軽量版**: 即座に起動  
- **完全版**: 初回モデルロード 60-120秒
- **動画処理**: ~1-2x リアルタイム（720p）
- **メモリ使用量**: 4-12GB

## 🔐 セキュリティ考慮事項

### ファイアウォール設定
```bash
# ローカルアクセスのみ許可（デフォルト）
# 外部アクセスが必要な場合のみポート開放
sudo pfctl -f /etc/pf.conf
```

### 一時ファイル管理
```bash
# 自動クリーンアップ設定
echo "cleanup_temp_files: true" >> config.yaml
echo "temp_dir: /tmp/bird_monitor" >> config.yaml
```

## 📚 参考資料

- [TensorFlow macOS Guide](https://www.tensorflow.org/install/pip)
- [PyTorch Apple Silicon](https://pytorch.org/get-started/locally/)
- [OpenCV Installation](https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html)
- [Homebrew Documentation](https://docs.brew.sh/)

## 🆘 サポート

問題が発生した場合：

1. **環境テスト実行**: `python test_macos.py`
2. **ログ確認**: `logs/` ディレクトリ内のログファイル
3. **Issue報告**: [GitHub Issues](https://github.com/atsuki-ichikawa/bird-monitoring-pipeline/issues)
4. **デバッグモード**: `LOG_LEVEL=DEBUG python -m src.web.app_minimal`

## 🎉 セットアップ完了後

軽量版が正常に動作することを確認したら：

1. **ブラウザアクセス**: http://localhost:8000
2. **API ドキュメント**: http://localhost:8000/docs  
3. **サンプル動画テスト**: `test_video.mp4` をアップロード
4. **完全版への移行**: ML依存関係インストール後

これでmacOS環境でのBird Monitoring Pipelineのセットアップは完了です！