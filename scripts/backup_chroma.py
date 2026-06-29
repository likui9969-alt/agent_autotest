"""
Chroma 向量库备份/恢复脚本

用法:
    python scripts/backup_chroma.py backup [name]    # 创建备份
    python scripts/backup_chroma.py restore <file>   # 从备份恢复
    python scripts/backup_chroma.py list              # 列出备份
    python scripts/backup_chroma.py clean <days>      # 清理 N 天前的备份

示例:
    python scripts/backup_chroma.py backup            # 使用时间戳命名
    python scripts/backup_chroma.py backup pre-release-v2
    python scripts/backup_chroma.py restore backups/chroma_pre-release-v2_2026-06-28.tar.gz
    python scripts/backup_chroma.py list
    python scripts/backup_chroma.py clean 30           # 删除 30 天前的备份
"""
import argparse
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import get_settings


def get_paths():
    """获取 Chroma 目录和备份目录"""
    settings = get_settings()
    chroma_dir = settings.get_chroma_dir()
    backup_dir = settings.get_backup_dir()
    return chroma_dir, backup_dir


def cmd_backup(args):
    """创建 Chroma 向量库备份"""
    chroma_dir, backup_dir = get_paths()

    if not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir):
        print(f"[错误] Chroma 目录不存在或为空: {chroma_dir}")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    name = args.name or f"chroma_auto_{timestamp}"
    backup_name = f"{name}_{timestamp}.tar.gz"
    backup_path = os.path.join(backup_dir, backup_name)

    print(f"创建备份: {backup_path}")
    print(f"  源目录: {chroma_dir}")

    # 创建压缩包
    start = time.time()
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(chroma_dir, arcname=os.path.basename(chroma_dir))
    elapsed = time.time() - start

    # 获取大小
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)

    print(f"  完成: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return 0


def cmd_restore(args):
    """从备份文件恢复 Chroma 向量库"""
    chroma_dir, _ = get_paths()
    backup_path = args.file

    if not os.path.isfile(backup_path):
        print(f"[错误] 备份文件不存在: {backup_path}")
        return 1

    if not backup_path.endswith(".tar.gz"):
        print(f"[错误] 备份文件格式不正确（需要 .tar.gz）: {backup_path}")
        return 1

    print(f"从备份恢复: {backup_path}")
    print(f"  目标目录: {chroma_dir}")

    # 备份现有目录
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        backup_dir = os.path.join(
            os.path.dirname(chroma_dir),
            f"chroma_before_restore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        )
        print(f"  备份当前数据到: {backup_dir}")
        shutil.copytree(chroma_dir, backup_dir)

    # 清除并恢复
    if os.path.isdir(chroma_dir):
        shutil.rmtree(chroma_dir)
    os.makedirs(chroma_dir, exist_ok=True)

    start = time.time()
    with tarfile.open(backup_path, "r:gz") as tar:
        tar.extractall(path=os.path.dirname(chroma_dir))
    elapsed = time.time() - start

    print(f"  完成 ({elapsed:.1f}s)")
    return 0


def cmd_list(args):
    """列出所有备份文件"""
    _, backup_dir = get_paths()

    if not os.path.isdir(backup_dir):
        print("备份目录不存在")
        return 0

    backups = sorted(
        [f for f in os.listdir(backup_dir) if f.endswith(".tar.gz")],
        reverse=True,
    )

    if not backups:
        print("暂无备份")
        return 0

    print(f"{'文件名':<60} {'大小':>10} {'创建时间'}")
    print("-" * 85)
    for name in backups:
        path = os.path.join(backup_dir, name)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        print(f"{name:<60} {size_mb:>8.1f}MB  {mtime.strftime('%Y-%m-%d %H:%M')}")

    print(f"\n总计: {len(backups)} 个备份")
    return 0


def cmd_clean(args):
    """清理 N 天前的旧备份"""
    _, backup_dir = get_paths()
    days = args.days
    cutoff = time.time() - days * 86400

    if not os.path.isdir(backup_dir):
        print("备份目录不存在")
        return 0

    removed = 0
    for f in os.listdir(backup_dir):
        path = os.path.join(backup_dir, f)
        if f.endswith(".tar.gz") and os.path.getmtime(path) < cutoff:
            os.remove(path)
            removed += 1
            print(f"  删除: {f}")

    print(f"清理完成: 删除 {removed} 个旧备份")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Chroma 向量库备份/恢复工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # backup
    p_backup = subparsers.add_parser("backup", help="创建备份")
    p_backup.add_argument("name", nargs="?", default="", help="备份名称（可选）")

    # restore
    p_restore = subparsers.add_parser("restore", help="从备份恢复")
    p_restore.add_argument("file", help="备份文件路径")

    # list
    subparsers.add_parser("list", help="列出所有备份")

    # clean
    p_clean = subparsers.add_parser("clean", help="清理旧备份")
    p_clean.add_argument("days", type=int, default=30, nargs="?", help="保留天数（默认 30）")

    args = parser.parse_args()

    if args.command == "backup":
        return cmd_backup(args)
    elif args.command == "restore":
        return cmd_restore(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "clean":
        return cmd_clean(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
