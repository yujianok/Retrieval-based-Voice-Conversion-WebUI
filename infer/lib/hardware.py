import hashlib
import subprocess
import re
import psutil
import winreg
import win32security
import win32api
import win32con

def get_hardware_fingerprint():
    """
    生成包含 Windows Machine SID 的硬件指纹
    - 包含完整 GPU 型号
    - 保留所有主板/BIOS 值
    """
    components = []

    # ==============================
    # 1. 主板序列号（保留所有值）
    # ==============================
    board_sn = _run_wmic_command('baseboard', 'SerialNumber')
    if board_sn:
        clean = board_sn.strip().upper()
        components.append(f"BOARD:{clean}")

    # ==============================
    # 2. BIOS 序列号（保留所有值）
    # ==============================
    bios_sn = _run_wmic_command('bios', 'SerialNumber')
    if bios_sn:
        clean = bios_sn.strip().upper()
        components.append(f"BIOS:{clean}")

    # ==============================
    # 3. 硬盘序列号
    # ==============================
    disk_sn = _run_wmic_command('diskdrive', 'SerialNumber')
    if disk_sn and len(disk_sn) >= 8:
        clean = re.sub(r'[^A-Z0-9]', '', disk_sn.upper())
        if len(clean) >= 8:
            components.append(f"DISK:{clean}")

    # ==============================
    # 4. CPU 物理核心数
    # ==============================
    cpu_cores = psutil.cpu_count(logical=False)
    if cpu_cores:
        components.append(f"CPU_CORES:{cpu_cores}")

    # ==============================
    # 5. 内存总容量（GB）
    # ==============================
    total_ram_gb = psutil.virtual_memory().total // (1024**3)
    components.append(f"RAM_GB:{total_ram_gb}")

    # ==============================
    # 6. GPU 型号（完整型号）
    # ==============================
    gpu_name = get_gpu_model()
    if gpu_name:
        # 清理特殊字符，保留字母、数字、空格、连字符
        clean_gpu = re.sub(r'[^A-Za-z0-9\s\-]', '', gpu_name)
        # 只保留关键部分，比如 "NVIDIA GeForce RTX 3060"
        if len(clean_gpu) >= 5:
            components.append(f"GPU:{clean_gpu.strip()}")

    # ==============================
    # 7. 系统制造商
    # ==============================
    manufacturer = _run_wmic_command('computersystem', 'Manufacturer')
    if manufacturer:
        manu_clean = re.sub(r'[^A-Za-z]', '', manufacturer.split()[0])
        if manu_clean and len(manu_clean) >= 3:
            components.append(f"MANU:{manu_clean.upper()}")

    # ==============================
    # 8. 产品型号
    # ==============================
    model = _run_wmic_command('computersystem', 'Model')
    if model:
        short_model = re.sub(r'[^A-Za-z0-9]', '', model.split()[0])[:8]
        if len(short_model) >= 3:
            components.append(f"MODEL:{short_model.upper()}")

    # ==============================
    # 9. CPU 品牌
    # ==============================
    cpu_name = _run_wmic_command('cpu', 'Name')
    if cpu_name:
        for key in ['Intel', 'AMD', 'i3', 'i5', 'i7', 'i9', 'Ryzen', 'Xeon']:
            if key.lower() in cpu_name.lower():
                components.append(f"CPU:{key.upper()}")
                break

    # ==============================
    # 10. Windows Machine SID
    # ==============================
    machine_sid = get_windows_machine_sid_pywin32()
    if machine_sid:
        components.append(f"SID:{machine_sid}")

    # ==============================
    # 生成最终 checksum
    # ==============================
    if not components:
        import uuid
        components = [f"FALLBACK:{str(uuid.uuid4())[:8]}"]

    unique_components = sorted(set(components))
    fingerprint_str = "|".join(unique_components)
    checksum = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()

    return {
        "fingerprint": fingerprint_str,
        "checksum": checksum,
        "components": unique_components
    }

def _run_wmic_command(class_name, property_name):
    try:
        cmd = f'wmic {class_name} get {property_name} /value'
        result = subprocess.run(
            cmd, shell=True,
            capture_output=True, text=True, timeout=5,
            encoding='utf-8', errors='ignore'
        )
        output = result.stdout.strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            if '=' in line:
                _, value = line.split('=', 1)
                if value.strip() and not value.startswith('NULL'):
                    return value.strip()
    except:
        pass
    return None

def get_gpu_model():
    """
    获取完整的 GPU 型号名称
    """
    try:
        cmd = 'wmic path win32_videocontroller get name /value'
        result = subprocess.run(
            cmd, shell=True,
            capture_output=True, text=True, timeout=5,
            encoding='utf-8', errors='ignore'
        )
        output = result.stdout.strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            if 'name=' in line.lower():
                _, value = line.split('=', 1)
                if value.strip() and not value.startswith('NULL'):
                    return value.strip()
    except:
        pass
    return None

def get_current_user_sid():
    try:
        process_token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), 
            win32con.TOKEN_READ
        )
        user_info = win32security.GetTokenInformation(
            process_token, 
            win32security.TokenUser
        )
        sid = user_info[0]
        sid_string = win32security.ConvertSidToStringSid(sid)
        return sid_string
    except:
        return None

def get_windows_machine_sid_pywin32():
    user_sid = get_current_user_sid()
    if user_sid and user_sid.startswith('S-1-5-21-'):
        parts = user_sid.split('-')
        if len(parts) >= 5:
            machine_sid = '-'.join(parts[:-1])
            return machine_sid.upper()
    return None

if __name__ == "__main__":
    try:
        result = get_hardware_fingerprint()
        print("🔧 Windows 硬件 + 系统指纹组件:")
        print(result['fingerprint'])
        print("\n🔐 SHA-256 Checksum:")
        print(result['checksum'])
    except ImportError as e:
        print(f"❌ 请先安装依赖: pip install pywin32 psutil")
    except Exception as e:
        print(f"❌ 错误: {e}")