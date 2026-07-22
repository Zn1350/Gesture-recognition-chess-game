# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# 获取虚拟环境路径
venv_path = os.path.join(os.path.abspath('.'), 'venv')
mediapipe_path = os.path.join(venv_path, 'Lib', 'site-packages', 'mediapipe')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # 模型文件
        ('model_improved.pth', '.'),
        ('micro_dual.pth', '.'),
        ('micro_dual', 'micro_dual'),
        # mediapipe 模型文件（手势识别必需）
        (os.path.join(mediapipe_path, 'modules'), 'mediapipe/modules'),
    ],
    hiddenimports=[
        # 项目核心依赖
        'Flask',
        'numpy',
        'numpy.core.multiarray',
        'numpy.core.umath',
        'mediapipe',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        'cv2',
        'pygame',
        'torch',
        'torch.nn',
        'torch.nn.functional',
        # mediapipe 依赖
        'matplotlib',
        'matplotlib.pyplot',
        'six',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大型库
        'scipy',
        'jax',
        'jaxlib',
        'pandas',
        'tensorflow',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RehabChess',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'python312.dll',
    ],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)