#!/usr/bin/env python3
"""
macOS用セットアップスクリプト
Apple Silicon及びIntel Mac対応
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

class MacOSSetup:
    def __init__(self):
        self.arch = platform.machine()
        self.python_version = sys.version_info
        self.is_apple_silicon = self.arch == "arm64"
        
    def check_requirements(self):
        """システム要件をチェック"""
        print("🔍 システム要件をチェック中...")
        
        # Python バージョンチェック
        if self.python_version >= (3, 13):
            print("⚠️  警告: Python 3.13はTensorFlowで完全サポートされていません")
            print("📝 Python 3.11または3.12の使用を推奨します")
            
            response = input("続行しますか？ (y/N): ")
            if response.lower() != 'y':
                print("❌ セットアップを中止しました")
                sys.exit(1)
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}")
        print(f"✅ アーキテクチャ: {self.arch}")
        
        # Homebrewチェック
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("✅ Homebrew インストール済み")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Homebrewが見つかりません")
            self.install_homebrew()
    
    def install_homebrew(self):
        """Homebrewをインストール"""
        print("📦 Homebrewをインストール中...")
        install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        os.system(install_cmd)
    
    def install_system_dependencies(self):
        """システム依存関係をインストール"""
        print("🔧 システム依存関係をインストール中...")
        
        deps = [
            "ffmpeg",           # 動画処理
            "portaudio",        # 音声処理
            "libsndfile",       # 音声ファイル処理
            "pkg-config",       # ビルドツール
        ]
        
        for dep in deps:
            try:
                print(f"  📦 {dep}をインストール中...")
                subprocess.run(["brew", "install", dep], check=True, capture_output=True)
                print(f"  ✅ {dep}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  {dep}のインストールに失敗: {e}")
    
    def setup_virtual_environment(self):
        """仮想環境をセットアップ"""
        print("🐍 Python仮想環境をセットアップ中...")
        
        venv_path = Path("venv")
        if venv_path.exists():
            print("  ℹ️  既存の仮想環境を削除中...")
            import shutil
            shutil.rmtree(venv_path)
        
        # 仮想環境作成
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("  ✅ 仮想環境作成完了")
        
        # アクティベーション方法を表示
        print("\n📝 仮想環境のアクティベーション方法:")
        print("   source venv/bin/activate")
    
    def install_python_dependencies(self):
        """Python依存関係をインストール"""
        print("📦 Python依存関係をインストール中...")
        
        # pip アップグレード
        pip_path = "venv/bin/pip"
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # 段階的インストール
        if self.is_apple_silicon:
            print("  🍎 Apple Silicon用の依存関係をインストール中...")
            self.install_apple_silicon_deps(pip_path)
        else:
            print("  💻 Intel Mac用の依存関係をインストール中...")
            self.install_intel_deps(pip_path)
    
    def install_apple_silicon_deps(self, pip_path):
        """Apple Silicon用依存関係"""
        # 基本依存関係
        subprocess.run([pip_path, "install", "-r", "requirements-minimal.txt"], check=True)
        
        # TensorFlow (Apple Silicon専用)
        print("  🤖 TensorFlow (Apple Silicon版)...")
        subprocess.run([pip_path, "install", "tensorflow-macos==2.13.0"], check=True)
        subprocess.run([pip_path, "install", "tensorflow-metal==1.0.1"], check=True)
        
        # PyTorch (Apple Silicon最適化)
        print("  🔥 PyTorch (Apple Silicon版)...")
        subprocess.run([pip_path, "install", "torch", "torchvision", "torchaudio"], check=True)
        
        # OpenCV
        print("  👁️  OpenCV...")
        subprocess.run([pip_path, "install", "opencv-python==4.8.1.78"], check=True)
        
        # その他のML依存関係
        print("  📊 その他のML依存関係...")
        ml_deps = [
            "ultralytics==8.0.196",
            "librosa==0.10.1",
            "scikit-learn==1.3.2",
            "numpy==1.24.3",
            "scipy==1.11.4",
            "pandas==2.1.4",
            "matplotlib==3.8.2"
        ]
        
        for dep in ml_deps:
            try:
                subprocess.run([pip_path, "install", dep], check=True)
                print(f"    ✅ {dep}")
            except subprocess.CalledProcessError:
                print(f"    ⚠️  {dep} - 後でインストールしてください")
    
    def install_intel_deps(self, pip_path):
        """Intel Mac用依存関係"""
        try:
            subprocess.run([pip_path, "install", "-r", "requirements-macos.txt"], check=True)
            print("  ✅ 全依存関係インストール完了")
        except subprocess.CalledProcessError:
            print("  ⚠️  一部の依存関係のインストールに失敗しました")
            print("  📝 手動でインストールが必要な場合があります")
    
    def test_installation(self):
        """インストールをテスト"""
        print("🧪 インストールをテスト中...")
        
        python_path = "venv/bin/python"
        
        # 基本モジュールテスト
        test_modules = [
            "fastapi",
            "uvicorn", 
            "pydantic",
            "numpy",
            "cv2",
        ]
        
        for module in test_modules:
            try:
                subprocess.run([python_path, "-c", f"import {module}"], 
                             check=True, capture_output=True)
                print(f"  ✅ {module}")
            except subprocess.CalledProcessError:
                print(f"  ❌ {module} - インポートに失敗")
        
        # TensorFlowテスト（Apple Siliconのみ）
        if self.is_apple_silicon:
            try:
                subprocess.run([python_path, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"], 
                             check=True, capture_output=True)
                print("  ✅ TensorFlow (Apple Silicon)")
            except subprocess.CalledProcessError:
                print("  ⚠️  TensorFlow - 後で手動インストールが必要かもしれません")
    
    def create_launch_script(self):
        """起動スクリプトを作成"""
        print("🚀 起動スクリプトを作成中...")
        
        launch_script = """#!/bin/bash
# Bird Monitoring Pipeline 起動スクリプト (macOS)

echo "🐦 Bird Monitoring Pipeline を起動中..."

# 仮想環境をアクティベート
source venv/bin/activate

# 軽量版から開始
echo "軽量版で起動します（テスト用）..."
python -m src.web.app_minimal &

echo "Web インターフェースが起動しました"
echo "ブラウザで http://localhost:8000 にアクセスしてください"
echo ""
echo "完全版を使用する場合:"
echo "  source venv/bin/activate"
echo "  python -m src.web.app"
echo ""
echo "停止するには Ctrl+C を押してください"

wait
"""
        
        launch_path = Path("start_macos.sh")
        launch_path.write_text(launch_script)
        launch_path.chmod(0o755)
        
        print(f"  ✅ {launch_path} を作成しました")
    
    def show_summary(self):
        """セットアップ完了サマリー"""
        print("\n🎉 macOS セットアップ完了!")
        print("\n📋 次のステップ:")
        print("1. 仮想環境をアクティベート:")
        print("   source venv/bin/activate")
        print("\n2. アプリケーションを起動:")
        print("   ./start_macos.sh")
        print("   または")
        print("   python -m src.web.app_minimal  # 軽量版")
        print("   python -m src.web.app          # 完全版")
        print("\n3. ブラウザでアクセス:")
        print("   http://localhost:8000")
        
        if self.python_version >= (3, 13):
            print("\n⚠️  注意:")
            print("Python 3.13を使用しています。一部のML機能で問題が発生する可能性があります。")
            print("完全な互換性のためにPython 3.11または3.12の使用を推奨します。")
    
    def run_setup(self):
        """完全セットアップを実行"""
        print("🍎 Bird Monitoring Pipeline - macOS セットアップ")
        print("=" * 50)
        
        try:
            self.check_requirements()
            self.install_system_dependencies()
            self.setup_virtual_environment()
            self.install_python_dependencies()
            self.test_installation()
            self.create_launch_script()
            self.show_summary()
            
        except KeyboardInterrupt:
            print("\n❌ セットアップが中断されました")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ セットアップエラー: {e}")
            print("📝 手動でのセットアップが必要かもしれません")
            sys.exit(1)

def main():
    setup = MacOSSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()