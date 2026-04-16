"""文件保护器 — 限制工作目录、自动备份、支持回滚."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


# 默认敏感文件模式（只读保护）
SENSITIVE_FILE_PATTERNS: list[str] = [
    ".env",
    "secrets.*",
    "*.pem",
    "*.key",
]


@dataclass
class FileModification:
    """记录一次文件修改."""

    path: str
    backup_path: str | None  # 备份文件路径（新建文件无备份）
    timestamp: float
    was_new: bool  # 是否是新创建的文件


class FileGuard:
    """文件保护器.

    - 限制 Agent 只能操作指定工作目录内的文件
    - 写操作前自动备份到 .agent-backups/
    - 记录所有修改历史，支持回滚
    - 敏感文件默认只读保护
    """

    def __init__(
        self,
        work_dir: str | Path,
        backup_dir_name: str = ".agent-backups",
        sensitive_patterns: list[str] | None = None,
    ) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.backup_dir = self.work_dir / backup_dir_name
        self.sensitive_patterns = sensitive_patterns or SENSITIVE_FILE_PATTERNS
        self._modifications: list[FileModification] = []

    def is_path_allowed(self, path: str | Path) -> bool:
        """检查路径是否在工作目录内."""
        try:
            resolved = Path(path).resolve()
            # 必须在工作目录内（或就是工作目录本身）
            return resolved == self.work_dir or self.work_dir in resolved.parents
        except (OSError, ValueError):
            return False

    def is_sensitive_file(self, path: str | Path) -> bool:
        """检查是否为敏感文件（.env, *.pem 等）."""
        p = Path(path)
        name = p.name

        for pattern in self.sensitive_patterns:
            if "*" in pattern:
                # 通配符匹配，如 *.pem
                import fnmatch
                if fnmatch.fnmatch(name, pattern):
                    return True
            else:
                # 精确匹配，如 .env
                if name == pattern:
                    return True
                # 前缀匹配，如 secrets.yaml 匹配 secrets.*
                if pattern.endswith(".*") and name.startswith(pattern[:-2] + "."):
                    return True
        return False

    def check_write(self, path: str | Path) -> tuple[bool, str]:
        """检查是否允许写入.

        Returns:
            (allowed, reason) — allowed=False 时 reason 说明原因
        """
        if not self.is_path_allowed(path):
            return False, f"路径不在工作目录内: {path}（工作目录: {self.work_dir}）"
        if self.is_sensitive_file(path):
            return False, f"敏感文件禁止写入: {Path(path).name}"
        return True, ""

    def check_read(self, path: str | Path) -> tuple[bool, str]:
        """检查是否允许读取.

        读取限制比写入宽松：只检查路径是否在工作目录内。
        """
        if not self.is_path_allowed(path):
            return False, f"路径不在工作目录内: {path}（工作目录: {self.work_dir}）"
        return True, ""

    def backup_file(self, path: str | Path) -> str | None:
        """在写入前备份文件.

        Returns:
            备份文件路径，如果文件不存在则返回 None
        """
        p = Path(path).resolve()
        if not p.exists():
            return None

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 备份文件名：原始名_时间戳
        timestamp = int(time.time() * 1000)
        # 保留相对路径结构，用 -- 替换 /
        rel = p.relative_to(self.work_dir)
        backup_name = str(rel).replace("/", "--") + f".{timestamp}.bak"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(str(p), str(backup_path))
        return str(backup_path)

    def record_modification(self, path: str | Path, backup_path: str | None) -> None:
        """记录一次文件修改."""
        p = Path(path).resolve()
        self._modifications.append(
            FileModification(
                path=str(p),
                backup_path=backup_path,
                timestamp=time.time(),
                was_new=backup_path is None,
            )
        )

    def pre_write(self, path: str | Path) -> tuple[bool, str]:
        """写入前的完整流程：检查权限 + 备份.

        Returns:
            (allowed, reason_or_backup_path)
        """
        allowed, reason = self.check_write(path)
        if not allowed:
            return False, reason

        backup_path = self.backup_file(path)
        self.record_modification(path, backup_path)
        return True, backup_path or ""

    def rollback_last(self) -> tuple[bool, str]:
        """回滚最近一次修改.

        Returns:
            (success, message)
        """
        if not self._modifications:
            return False, "没有可回滚的修改"

        mod = self._modifications.pop()
        return self._rollback_one(mod)

    def rollback_all(self) -> list[tuple[bool, str]]:
        """回滚所有修改（按逆序）.

        Returns:
            每次回滚的 (success, message) 列表
        """
        results: list[tuple[bool, str]] = []
        while self._modifications:
            mod = self._modifications.pop()
            results.append(self._rollback_one(mod))
        return results

    @property
    def modifications(self) -> list[FileModification]:
        """返回所有修改记录（只读）."""
        return list(self._modifications)

    def _rollback_one(self, mod: FileModification) -> tuple[bool, str]:
        """回滚单个修改."""
        target = Path(mod.path)

        if mod.was_new:
            # 新建的文件 → 删除
            if target.exists():
                target.unlink()
                return True, f"已删除新建文件: {mod.path}"
            return True, f"文件已不存在: {mod.path}"

        # 有备份 → 恢复
        if mod.backup_path and Path(mod.backup_path).exists():
            shutil.copy2(mod.backup_path, mod.path)
            return True, f"已从备份恢复: {mod.path}"

        return False, f"备份文件不存在，无法恢复: {mod.path}"
