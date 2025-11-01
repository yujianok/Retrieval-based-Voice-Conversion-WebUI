# pyinstaller_hooks/hook-infer.lib.infer_pack.py
from PyInstaller.utils.hooks import collect_data_files

# ✅ 关键：收集 infer.lib.infer_pack 包下的所有文件，包括 .py 源文件
# 这会包含 commons.py, models.py 等
datas = collect_data_files('infer.lib.infer_pack', include_py_files=True)