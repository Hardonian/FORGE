# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for packaging FORGE into a standalone distribution binary."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

repo_root = Path(SPECPATH).resolve().parent if "SPECPATH" in globals() else Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
forge_dir = repo_root / "forge"

# Collect all submodules across forge and third-party packages
hidden_imports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "pydantic",
    "pydantic_core",
    "starlette",
    "anyio",
    "typer",
    "rich",
]
hidden_imports += collect_submodules("forge")
hidden_imports += collect_submodules("pydantic")
hidden_imports += collect_submodules("pydantic_core")

# Collect sample models and data files
datas = [
    (str(repo_root / "examples"), "examples"),
    (str(repo_root / "README.md"), "."),
    (str(repo_root / "LICENSE"), "."),
]

# Collect any data files required by packages
datas += collect_data_files("pydantic")
datas += collect_data_files("fastapi")

a = Analysis(
    [str(forge_dir / "__main__.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "pytest",
        "IPython",
        "jupyter",
        "jupyterlab",
        "notebook",
        "tornado",
        "transformers",
        "accelerate",
        "grpcio",
        "google",
        "deepinfer_serving",
        "agentaxiom",
        "torch",
        "torchvision",
        "torchaudio",
        "rocm_sdk",
        "_rocm_sdk_core",
        "_rocm_sdk_libraries",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="forge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="forge",
)
