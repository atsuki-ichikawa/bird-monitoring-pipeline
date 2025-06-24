#!/usr/bin/env python3
"""
Windows実行ファイル作成スクリプト
段階的配布アプローチを使用
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import zipfile
import requests

class WindowsBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build_windows"
        self.dist_dir = self.project_root / "dist_windows"
        
    def create_launcher(self):
        """軽量ランチャーを作成"""
        launcher_code = '''
import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import threading
import requests
import zipfile

class BirdMonitorLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bird Monitoring Pipeline")
        self.root.geometry("600x400")
        
        self.setup_ui()
        self.check_installation()
    
    def setup_ui(self):
        # タイトル
        title = tk.Label(self.root, text="Bird Monitoring Pipeline", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=20)
        
        # 状態表示
        self.status_label = tk.Label(self.root, text="システムチェック中...")
        self.status_label.pack(pady=10)
        
        # プログレスバー
        self.progress = ttk.Progressbar(self.root, length=400, mode='indeterminate')
        self.progress.pack(pady=10)
        
        # ボタンフレーム
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.install_btn = tk.Button(button_frame, text="依存関係をインストール", 
                                   command=self.install_dependencies, state="disabled")
        self.install_btn.pack(side=tk.LEFT, padx=10)
        
        self.start_btn = tk.Button(button_frame, text="アプリケーション開始", 
                                 command=self.start_application, state="disabled")
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        # ログエリア
        self.log_text = tk.Text(self.root, height=10, width=70)
        self.log_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
    
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def check_installation(self):
        """インストール状態をチェック"""
        self.progress.start()
        threading.Thread(target=self._check_installation_thread, daemon=True).start()
    
    def _check_installation_thread(self):
        try:
            # Python依存関係チェック
            self.log("Python依存関係をチェック中...")
            missing_packages = self.check_python_packages()
            
            # モデルファイルチェック
            self.log("モデルファイルをチェック中...")
            missing_models = self.check_models()
            
            # 結果に基づいてUI更新
            if missing_packages or missing_models:
                self.status_label.config(text="セットアップが必要です")
                self.install_btn.config(state="normal")
            else:
                self.status_label.config(text="準備完了")
                self.start_btn.config(state="normal")
                
        except Exception as e:
            self.log(f"エラー: {e}")
            messagebox.showerror("エラー", f"チェック中にエラーが発生しました: {e}")
        finally:
            self.progress.stop()
    
    def check_python_packages(self):
        """必要なPythonパッケージをチェック"""
        required = ["fastapi", "uvicorn", "opencv-python", "tensorflow", "torch"]
        missing = []
        
        for package in required:
            try:
                __import__(package.replace("-", "_"))
                self.log(f"✓ {package}")
            except ImportError:
                self.log(f"✗ {package} (未インストール)")
                missing.append(package)
        
        return missing
    
    def check_models(self):
        """モデルファイルをチェック"""
        models_dir = Path("models")
        required_models = {
            "birdnet": "models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model.tflite",
            "yolo": "models/yolo/yolov10n.pt"
        }
        
        missing = []
        for name, path in required_models.items():
            if Path(path).exists():
                self.log(f"✓ {name} model")
            else:
                self.log(f"✗ {name} model (未ダウンロード)")
                missing.append(name)
        
        return missing
    
    def install_dependencies(self):
        """依存関係をインストール"""
        self.progress.start()
        self.install_btn.config(state="disabled")
        threading.Thread(target=self._install_dependencies_thread, daemon=True).start()
    
    def _install_dependencies_thread(self):
        try:
            # Python依存関係インストール
            self.log("Python依存関係をインストール中...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True, text=True)
            
            # モデルダウンロード
            self.log("モデルファイルをダウンロード中...")
            self.download_models()
            
            self.log("インストール完了!")
            self.status_label.config(text="準備完了")
            self.start_btn.config(state="normal")
            
        except Exception as e:
            self.log(f"インストールエラー: {e}")
            messagebox.showerror("エラー", f"インストール中にエラーが発生しました: {e}")
        finally:
            self.progress.stop()
    
    def download_models(self):
        """モデルファイルをダウンロード"""
        models_urls = {
            "yolo": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
        }
        
        Path("models/yolo").mkdir(parents=True, exist_ok=True)
        
        for name, url in models_urls.items():
            self.log(f"{name} モデルをダウンロード中...")
            response = requests.get(url, stream=True)
            filename = Path("models") / name / url.split('/')[-1]
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.log(f"✓ {name} モデルダウンロード完了")
    
    def start_application(self):
        """アプリケーションを開始"""
        try:
            self.log("Bird Monitoring Pipeline を開始中...")
            subprocess.Popen([sys.executable, "-m", "src.web.app"])
            self.log("アプリケーションが開始されました")
            self.log("ブラウザで http://localhost:8000 にアクセスしてください")
        except Exception as e:
            messagebox.showerror("エラー", f"アプリケーション開始エラー: {e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    launcher = BirdMonitorLauncher()
    launcher.run()
'''
        
        launcher_file = self.build_dir / "launcher.py"
        launcher_file.parent.mkdir(parents=True, exist_ok=True)
        launcher_file.write_text(launcher_code, encoding='utf-8')
        
        return launcher_file
    
    def create_installer_script(self):
        """Windows用インストーラースクリプトを作成"""
        installer_code = '''
@echo off
echo Bird Monitoring Pipeline インストーラー
echo =====================================

:: Python インストールチェック
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python がインストールされていません。
    echo https://www.python.org/downloads/ からPython 3.11をダウンロードしてください。
    pause
    exit /b 1
)

:: 仮想環境作成
echo 仮想環境を作成中...
python -m venv bird_monitor_env
call bird_monitor_env\\Scripts\\activate.bat

:: 基本依存関係インストール
echo 基本依存関係をインストール中...
pip install -r requirements-minimal.txt

:: ランチャー起動
echo ランチャーを起動中...
python launcher.py

pause
'''
        
        installer_file = self.build_dir / "install.bat"
        installer_file.write_text(installer_code, encoding='utf-8')
        
        return installer_file
    
    def build_distribution(self):
        """配布パッケージを作成"""
        print("Windows配布パッケージを作成中...")
        
        # ビルドディレクトリクリア
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True)
        
        # 必要ファイルをコピー
        print("ソースファイルをコピー中...")
        shutil.copytree(self.project_root / "src", self.build_dir / "src")
        shutil.copy2(self.project_root / "requirements.txt", self.build_dir)
        shutil.copy2(self.project_root / "requirements-minimal.txt", self.build_dir)
        shutil.copy2(self.project_root / "README.md", self.build_dir)
        
        # ランチャー作成
        print("ランチャーを作成中...")
        self.create_launcher()
        
        # インストーラー作成
        print("インストーラーを作成中...")
        self.create_installer_script()
        
        # README for Windows作成
        windows_readme = '''
# Bird Monitoring Pipeline - Windows版

## インストール手順

1. install.bat をダブルクリックして実行
2. Python依存関係が自動的にインストールされます
3. ランチャーが起動したら「依存関係をインストール」をクリック
4. 完了後「アプリケーション開始」をクリック

## 使用方法

- ブラウザで http://localhost:8000 にアクセス
- 動画ファイルをアップロードして解析開始

## システム要件

- Windows 10/11
- Python 3.11以降
- 8GB以上のRAM (推奨16GB)
- 2GB以上の空きディスク容量

## サポート

問題が発生した場合は、GitHub Issues を確認してください:
https://github.com/atsuki-ichikawa/bird-monitoring-pipeline/issues
'''
        
        (self.build_dir / "README_Windows.txt").write_text(windows_readme, encoding='utf-8')
        
        # ZIPファイル作成
        print("ZIPファイルを作成中...")
        self.dist_dir.mkdir(exist_ok=True)
        zip_path = self.dist_dir / "BirdMonitoringPipeline_Windows.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.build_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.build_dir)
                    zipf.write(file_path, arcname)
        
        print(f"配布パッケージが作成されました: {zip_path}")
        return zip_path

def main():
    builder = WindowsBuilder()
    zip_path = builder.build_distribution()
    print(f"\n✅ Windows配布パッケージ作成完了!")
    print(f"📦 ファイル: {zip_path}")
    print(f"📁 サイズ: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()