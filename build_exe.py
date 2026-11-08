"""Build the Gomoku desktop executable from the existing PyInstaller spec."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def build_exe():
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("请先用当前 Python 安装 PyInstaller：python -m pip install pyinstaller")

    root = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(root / "RehabChess.spec")],
        cwd=root,
        check=True,
    )
    name = "Gomoku.exe" if sys.platform == "win32" else "Gomoku"
    print(f"打包完成。可执行文件位于 {root / 'dist' / name}")


if __name__ == "__main__":
    build_exe()
