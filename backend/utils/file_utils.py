"""
通用文件工具模块
"""
import hashlib
from pathlib import Path


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """计算文件的哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5 / sha256)

    Returns:
        十六进制哈希字符串
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(filename: str) -> str:
    """清理文件名中的不安全字符

    Args:
        filename: 原始文件名

    Returns:
        安全的文件名
    """
    # 移除路径分隔符和其他危险字符
    unsafe_chars = '<>:"/\\|?*\''
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    # 限制长度
    if len(filename) > 200:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:200 - len(ext)] + ext
    return filename


def detect_file_type(file_path: str) -> str:
    """根据文件扩展名检测文件类型

    Args:
        file_path: 文件路径

    Returns:
        文件类型标识 (txt / pdf / docx / unknown)
    """
    ext = Path(file_path).suffix.lower()
    type_map = {
        ".txt": "txt",
        ".pdf": "pdf",
        ".docx": "docx",
        ".log": "txt",
        ".md": "txt",
    }
    return type_map.get(ext, "unknown")
