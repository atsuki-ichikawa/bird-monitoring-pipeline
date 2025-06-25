#!/usr/bin/env python3
"""
macOS環境テストスクリプト
現在の環境で利用可能な機能をチェック
"""

import sys
import platform
import subprocess
from pathlib import Path

def test_python_version():
    """Python バージョンテスト"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 13):
        print("⚠️  Python 3.13: TensorFlowサポートに制限あり")
        return "warning"
    elif version >= (3, 11):
        print("✅ Python バージョン: 完全対応")
        return "ok"
    else:
        print("❌ Python バージョン: 3.11以上が必要")
        return "error"

def test_architecture():
    """macOS アーキテクチャテスト"""
    arch = platform.machine()
    print(f"アーキテクチャ: {arch}")
    
    if arch == "arm64":
        print("🍎 Apple Silicon Mac - tensorflow-macos使用")
        return "apple_silicon"
    elif arch == "x86_64":
        print("💻 Intel Mac - 標準tensorflow使用")
        return "intel"
    else:
        print(f"❓ 不明なアーキテクチャ: {arch}")
        return "unknown"

def test_basic_imports():
    """基本モジュールインポートテスト"""
    modules = {
        "fastapi": "Web フレームワーク",
        "uvicorn": "ASGI サーバー",
        "pydantic": "データ検証",
        "pandas": "データ処理",
        "numpy": "数値計算"
    }
    
    results = {}
    for module, desc in modules.items():
        try:
            __import__(module)
            print(f"✅ {module}: {desc}")
            results[module] = True
        except ImportError:
            print(f"❌ {module}: {desc} (未インストール)")
            results[module] = False
    
    return results

def test_ml_imports():
    """ML関連モジュールテスト"""
    ml_modules = {
        "tensorflow": "機械学習フレームワーク",
        "cv2": "コンピュータビジョン",
        "torch": "PyTorch",
        "sklearn": "機械学習ライブラリ",
        "librosa": "音声処理"
    }
    
    results = {}
    for module, desc in ml_modules.items():
        try:
            if module == "cv2":
                import cv2
                print(f"✅ {module}: {desc} (v{cv2.__version__})")
            elif module == "tensorflow":
                import tensorflow as tf
                print(f"✅ {module}: {desc} (v{tf.__version__})")
            elif module == "torch":
                import torch
                print(f"✅ {module}: {desc} (v{torch.__version__})")
            elif module == "sklearn":
                import sklearn
                print(f"✅ {module}: {desc} (v{sklearn.__version__})")
            elif module == "librosa":
                import librosa
                print(f"✅ {module}: {desc} (v{librosa.__version__})")
            else:
                __import__(module)
                print(f"✅ {module}: {desc}")
            results[module] = True
        except ImportError:
            print(f"⚠️  {module}: {desc} (未インストール)")
            results[module] = False
        except Exception as e:
            print(f"❌ {module}: {desc} (エラー: {e})")
            results[module] = False
    
    return results

def test_system_dependencies():
    """システム依存関係テスト"""
    system_deps = {
        "ffmpeg": "動画処理",
        "brew": "パッケージマネージャー"
    }
    
    results = {}
    for cmd, desc in system_deps.items():
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}: {desc}")
                results[cmd] = True
            else:
                print(f"❌ {cmd}: {desc} (インストールされていません)")
                results[cmd] = False
        except FileNotFoundError:
            print(f"❌ {cmd}: {desc} (インストールされていません)")
            results[cmd] = False
    
    return results

def test_project_structure():
    """プロジェクト構造テスト"""
    required_paths = [
        "src/web/app_minimal.py",
        "src/cli_minimal.py", 
        "requirements-minimal.txt",
        "tests/test_minimal.py"
    ]
    
    results = {}
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path}")
            results[path_str] = True
        else:
            print(f"❌ {path} (見つかりません)")
            results[path_str] = False
    
    return results

def recommend_next_steps(test_results):
    """次のステップを推奨"""
    print("\n📋 推奨される次のステップ:")
    
    basic_modules = test_results.get("basic_imports", {})
    ml_modules = test_results.get("ml_imports", {})
    system_deps = test_results.get("system_deps", {})
    
    # 基本モジュールが不足している場合
    missing_basic = [m for m, available in basic_modules.items() if not available]
    if missing_basic:
        print("1. 基本依存関係をインストール:")
        print("   pip install -r requirements-minimal.txt")
        print("   pip install pydantic-settings")
    
    # ML モジュールが不足している場合
    missing_ml = [m for m, available in ml_modules.items() if not available]
    if missing_ml:
        print("2. ML依存関係をインストール（オプション）:")
        arch = test_results.get("arch")
        if arch == "apple_silicon":
            print("   # Apple Silicon用")
            print("   pip install tensorflow-macos tensorflow-metal")
        else:
            print("   # Intel Mac用")
            print("   pip install tensorflow")
        print("   pip install opencv-python torch torchvision")
    
    # システム依存関係が不足している場合
    if not system_deps.get("brew", True):
        print("3. Homebrewをインストール:")
        print('   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
    
    if not system_deps.get("ffmpeg", True):
        print("4. システム依存関係をインストール:")
        print("   brew install ffmpeg portaudio libsndfile")
    
    # 利用可能な起動方法
    print("5. アプリケーションを起動:")
    if basic_modules.get("fastapi", False):
        print("   python -m src.web.app_minimal  # 軽量版")
        if missing_ml:
            print("   # 完全版は ML依存関係インストール後に利用可能")
        else:
            print("   python -m src.web.app          # 完全版")

def main():
    print("🍎 Bird Monitoring Pipeline - macOS 環境テスト")
    print("=" * 60)
    
    results = {}
    
    print("\n📍 システム情報:")
    results["python"] = test_python_version()
    results["arch"] = test_architecture()
    
    print("\n📦 基本モジュール:")
    results["basic_imports"] = test_basic_imports()
    
    print("\n🤖 ML モジュール:")
    results["ml_imports"] = test_ml_imports()
    
    print("\n🔧 システム依存関係:")
    results["system_deps"] = test_system_dependencies()
    
    print("\n📁 プロジェクト構造:")
    results["project"] = test_project_structure()
    
    # サマリー
    print("\n" + "=" * 60)
    recommend_next_steps(results)
    
    print(f"\n🎯 現在の状況:")
    basic_ready = all(results["basic_imports"].values())
    ml_ready = all(results["ml_imports"].values())
    
    if basic_ready:
        print("✅ 軽量版: 利用可能")
        print("   python -m src.web.app_minimal")
    else:
        print("⚠️  軽量版: 依存関係不足")
    
    if ml_ready:
        print("✅ 完全版: 利用可能") 
        print("   python -m src.web.app")
    else:
        print("⚠️  完全版: ML依存関係不足")
    
    print(f"\n📖 詳細なセットアップガイド: BUILD_MACOS.md")

if __name__ == "__main__":
    main()