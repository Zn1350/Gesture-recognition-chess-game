import os
import subprocess
import sys

def build_exe():
    # 确保 PyInstaller 已安装
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 资源文件列表
    resources = [
        ("model_improved.pth", "."),
        ("micro_dual.pth", "."),
    ]
    
    # 构建 PyInstaller 命令
    cmd = [
        "pyinstaller",
        "--name=RehabChess",
        "--onefile",
        "--windowed",
        "--icon=chess.ico",  # 如果有图标文件的话
        "--add-data={};.".format("model_improved.pth"),
        "--add-data={};.".format("micro_dual.pth"),
        "--hidden-import=mediapipe",
        "--hidden-import=cv2",
        "--hidden-import=torch",
        "--hidden-import=pygame",
        "--hidden-import=numpy",
        "--paths={}".format(root_dir),
        "main.py"
    ]
    
    # 移除空参数（如图标文件不存在）
    cmd = [c for c in cmd if c != "--icon=chess.ico" or os.path.exists("chess.ico")]
    
    print("执行打包命令:", " ".join(cmd))
    
    # 执行打包
    subprocess.check_call(cmd, cwd=root_dir)
    
    print("\n✅ 打包完成！可执行文件位于 dist/RehabChess.exe")

if __name__ == "__main__":
    build_exe()