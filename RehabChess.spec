# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

root = Path(SPECPATH)
model_files = (
    "face_detection_yunet_2023mar.onnx",
    "facial_expression_recognition_mobilefacenet_2022july.onnx",
    "LICENSE-YuNet",
    "LICENSE-MobileFaceNet",
)

a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    datas=[(str(root / "models" / filename), "models") for filename in model_files],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Gomoku",
    console=False,
    upx=False,
)
