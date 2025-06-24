# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for Bird Monitoring Pipeline
Windows実行ファイル作成用
"""

import os
from pathlib import Path

# プロジェクトルート
project_root = Path(SPECPATH)
src_dir = project_root / 'src'

# 収集するデータファイル
datas = [
    # Web関連ファイル
    (str(src_dir / 'web' / 'templates'), 'templates'),
    (str(src_dir / 'web' / 'static'), 'static'),
    
    # 設定ファイル
    ('config.yaml', '.'),
    ('requirements-minimal.txt', '.'),
    
    # ドキュメント
    ('README.md', '.'),
]

# 隠しインポート（動的にインポートされるモジュール）
hiddenimports = [
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.protocols.http.h11_impl',
    'fastapi.routing',
    'fastapi.encoders',
    'pydantic.validators',
    'pydantic.json',
    'email_validator',
]

# 除外するモジュール（サイズ削減）
excludes = [
    'matplotlib',
    'scipy',
    'IPython',
    'notebook',
    'jupyter',
    'pandas.plotting',
    'pandas.tests',
    'numpy.tests',
    'test',
    'tests',
    'testing',
]

# ランチャー用の軽量版作成
launcher_a = Analysis(
    ['build_windows.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

launcher_pyz = PYZ(launcher_a.pure, launcher_a.zipped_data, cipher=None)

launcher_exe = EXE(
    launcher_pyz,
    launcher_a.scripts,
    launcher_a.binaries,
    launcher_a.zipfiles,
    launcher_a.datas,
    [],
    name='BirdMonitor_Launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI用
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)

# CLI版（軽量）
cli_minimal_a = Analysis(
    [str(src_dir / 'cli_minimal.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ('requirements-minimal.txt', '.'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'typer',
        'rich.console',
        'rich.table',
        'rich.progress',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

cli_minimal_pyz = PYZ(cli_minimal_a.pure, cli_minimal_a.zipped_data, cipher=None)

cli_minimal_exe = EXE(
    cli_minimal_pyz,
    cli_minimal_a.scripts,
    cli_minimal_a.binaries,
    cli_minimal_a.zipfiles,
    cli_minimal_a.datas,
    [],
    name='BirdMonitor_CLI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI用
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Webアプリ版（軽量）
web_minimal_a = Analysis(
    [str(src_dir / 'web' / 'app_minimal.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

web_minimal_pyz = PYZ(web_minimal_a.pure, web_minimal_a.zipped_data, cipher=None)

web_minimal_exe = EXE(
    web_minimal_pyz,
    web_minimal_a.scripts,
    web_minimal_a.binaries,
    web_minimal_a.zipfiles,
    web_minimal_a.datas,
    [],
    name='BirdMonitor_Web',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # デバッグ用にコンソール表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 全体をまとめたディストリビューション
coll = COLLECT(
    launcher_exe,
    cli_minimal_exe,
    web_minimal_exe,
    launcher_a.binaries,
    cli_minimal_a.binaries,
    web_minimal_a.binaries,
    launcher_a.zipfiles,
    cli_minimal_a.zipfiles,
    web_minimal_a.zipfiles,
    launcher_a.datas,
    cli_minimal_a.datas,
    web_minimal_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BirdMonitoringPipeline'
)