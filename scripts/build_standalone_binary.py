"""Automated cross-platform build and verification script for FORGE standalone distribution binaries."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from forge import __version__  # noqa: E402


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    spec_path = repo_root / "packaging" / "forge.spec"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    print("=================================================================")
    print(f"  FORGE Standalone Distribution Packager v{__version__}")
    print(f"  Host Platform: {platform.system()} {platform.machine()}")
    print("=================================================================\n")

    # 1. Clean previous build artifacts
    print("[1/5] Cleaning previous build artifacts...")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # 2. Execute PyInstaller build
    print("[2/5] Compiling standalone distribution binary with PyInstaller...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        str(spec_path),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    last_line = ""
    for line in proc.stdout:
        last_line = line.strip()
        if "INFO:" in line or "WARNING:" in line or "Building" in line:
            print(f"  > {last_line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] PyInstaller compilation failed with code {proc.returncode}!", file=sys.stderr)
        print(f"Last message: {last_line}", file=sys.stderr)
        return 1

    # Locate generated executable
    binary_name = "forge.exe" if platform.system() == "Windows" else "forge"
    bin_path = dist_dir / "forge" / binary_name
    if not bin_path.exists():
        print(f"[ERROR] Executable was not found at expected path: {bin_path}", file=sys.stderr)
        return 1

    print(f"[OK] Compiled standalone executable: {bin_path}")

    # 3. Smoke Test Compiled Executable
    print("[3/5] Performing self-test verification on compiled standalone binary...")
    ver_cmd = [str(bin_path), "--version"]
    ver_res = subprocess.run(ver_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if ver_res.returncode != 0 or "FORGE" not in ver_res.stdout:
        print(f"[ERROR] Binary failed '--version' check: {ver_res.stderr}", file=sys.stderr)
        return 1
    print(f"[OK] Binary responded: {ver_res.stdout.strip()}")

    # Test quick CLI execution
    demo_out = dist_dir / "smoke_test_demo"
    demo_cmd = [str(bin_path), "demo", "--out", str(demo_out)]
    demo_res = subprocess.run(demo_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if demo_res.returncode != 0:
        print(f"[ERROR] Binary failed 'demo' execution test: {demo_res.stderr}", file=sys.stderr)
        return 1
    print("[OK] Binary completed end-to-end hardware inverse design showcase.")

    # 4. Create Distribution Archive (.zip)
    print("[4/5] Packaging portable distribution ZIP archive...")
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    archive_name = f"forge-v{__version__}-{sys_name}-{machine}.zip"
    archive_path = dist_dir / archive_name

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in (dist_dir / "forge").rglob("*"):
            if file.is_file():
                rel = file.relative_to(dist_dir / "forge")
                zf.write(file, arcname=Path("forge") / rel)

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    archive_hash = sha256_file(archive_path)
    print(f"[OK] Created archive: {archive_path.name} ({archive_size_mb:.1f} MB)")

    # 5. Output Manifest
    print("\n[5/5] Standalone Distribution Artifacts:")
    print("-----------------------------------------------------------------")
    print(f"Archive:       {archive_path}")
    print(f"Size:          {archive_size_mb:.2f} MB")
    print(f"SHA-256:       {archive_hash}")
    print("-----------------------------------------------------------------")
    print("\n[SUCCESS] Standalone distribution binary ready for deployment!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
