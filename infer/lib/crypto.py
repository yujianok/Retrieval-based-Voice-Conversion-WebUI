from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii
import os
import sys

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.hardware import get_hardware_fingerprint

def load_public_key_from_file(file_path):
    """
    从文件系统读取RSA公钥
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"公钥文件不存在: {file_path}")
    
    with open(file_path, 'rb') as f:
        public_key_pem = f.read()
    
    # 验证是否为有效的RSA公钥
    try:
        RSA.import_key(public_key_pem)
    except ValueError:
        raise ValueError("文件内容不是有效的RSA公钥")
    
    return public_key_pem

def encrypt_with_public_key(fingerprint, public_key_pem):
    """
    使用公钥加密SID
    """
    # 将SID字符串编码为字节
    sid_bytes = fingerprint.encode('utf-8')
    
    # 导入公钥
    rsa_key = RSA.import_key(public_key_pem)
    cipher = PKCS1_OAEP.new(rsa_key)
    
    # 加密SID字节
    encrypted_bytes = cipher.encrypt(sid_bytes)
    
    # 转换为16进制格式
    encrypted_hex = binascii.hexlify(encrypted_bytes).decode('utf-8')
    return encrypted_hex

def device_fingerprint():
    fingerprint = get_hardware_fingerprint()['checksum']
    file_path = os.path.join(os.getcwd(), 'public_key.pem')
    public_key_pem = load_public_key_from_file(file_path)
    encrypted_sid_hex = encrypt_with_public_key(fingerprint, public_key_pem)

    return encrypted_sid_hex


def derive_key_from_password(password, key_length=32):
    """
    从密码派生AES密钥
    """
    # 使用SHA-256哈希函数派生密钥
    hash_obj = hashlib.sha256(password.encode('utf-8'))
    return hash_obj.digest()[:key_length]

def decrypt_file(encrypted_file_path):
    """
    使用设备指纹作为密码解密单个文件，返回解密后的数据在内存中的缓冲区
    """
    # 获取设备指纹作为密码
    password = get_hardware_fingerprint()['checksum']
    
    # 从密码派生AES密钥
    key = derive_key_from_password(password)
    
    # 读取加密文件内容
    with open(encrypted_file_path, 'rb') as f:
        encrypted_content = f.read()
    
    # 提取IV（前16字节）和实际加密数据
    iv = encrypted_content[:16]
    encrypted_data = encrypted_content[16:]
    
    # 创建AES解密器，使用相同的IV
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # 解密数据
    padded_data = cipher.decrypt(encrypted_data)
    
    # 移除填充
    file_data = unpad(padded_data, AES.block_size)
    
    # 返回解密后的数据缓冲区
    return file_data



