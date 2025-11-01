# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['api_240604.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/hubert', 'assets/hubert'), ('assets/rmvpe', 'assets/rmvpe'), ('assets/uvr5_weights', 'assets/uvr5_weights'), ('assets/Synthesizer_inputs.pth', 'assets/Synthesizer_inputs.pth'), ('configs', 'configs'), ('i18n', 'i18n'), ('logs', 'logs'), ('ffmpeg.exe', '.'), ('ffprobe.exe', '.'), ('public_key.pem', '.')],
    hiddenimports=[],
    hookspath=['pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='api_240604',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='api_240604',
)
