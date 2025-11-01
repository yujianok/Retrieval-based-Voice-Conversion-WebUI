rmdir /s /q dist
rmdir /s /q build
rmdir /s /q __pycache__
rmdir /s /q pyinstaller_hooks\__pycache__

rmdir /s /q "%LOCALAPPDATA%\pyinstaller"

runtime\python.exe -m PyInstaller ^
    --additional-hooks-dir "pyinstaller_hooks" ^
    --add-data "assets/hubert;assets/hubert" ^
    --add-data "assets/rmvpe;assets/rmvpe" ^
    --add-data "assets/uvr5_weights;assets/uvr5_weights" ^
    --add-data "assets/Synthesizer_inputs.pth;assets/" ^
    --add-data "configs;configs" ^
    --add-data "i18n;i18n" ^
    --add-data "logs;logs" ^
    --add-data "ffmpeg.exe;." ^
    --add-data "ffprobe.exe;." ^
    --add-data "public_key.pem;." ^
    api_240604.py