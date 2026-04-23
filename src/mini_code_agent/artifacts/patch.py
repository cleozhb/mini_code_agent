"""Patch 数据结构 — 描述一组代码变更及其应用逻辑."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EditOperation(str, Enum):
    """文件编辑操作类型."""

    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    RENAME = "RENAME"


@dataclass
class FileEdit:
    """单个文件的编辑操作."""

    path: str  # 相对于项目根的路径
    operation: EditOperation  # CREATE / MODIFY / DELETE / RENAME
    old_content: str | None  # 修改前的完整内容（CREATE 时为 None）
    new_content: str | None  # 修改后的完整内容（DELETE 时为 None）
    old_path: str | None  # 仅 RENAME 时使用
    unified_diff: str  # 标准 unified diff 格式，用于展示
    lines_added: int
    lines_removed: int


@dataclass
class ApplyResult:
    """Patch 应用结果."""

    success: bool
    applied_files: list[str] = field(default_factory=list)  # 成功应用的文件
    failed_files: list[str] = field(default_factory=list)  # 失败的文件
    errors: list[str] = field(default_factory=list)  # 错误信息


@dataclass(frozen=True)
class Patch:
    """一组文件编辑组成的 Patch."""

    edits: list[FileEdit]
    total_files_changed: int
    total_lines_added: int
    total_lines_removed: int
    base_git_hash: str  # 基于哪个 commit 产生的 patch

    def to_unified_diff(self) -> str:
        """合并所有 edits 成一个大 diff."""
        parts: list[str] = []
        for edit in self.edits:
            if edit.unified_diff:
                parts.append(edit.unified_diff)
        return "\n".join(parts)

    def apply_to(self, project_path: str) -> ApplyResult:
        """在目标项目上应用这个 patch，返回成功/失败信息."""
        result = ApplyResult(success=True)
        root = Path(project_path)

        for edit in self.edits:
            target = root / edit.path
            try:
                if edit.operation == EditOperation.CREATE:
                    if target.exists():
                        result.failed_files.append(edit.path)
                        result.errors.append(
                            f"CREATE 失败: 文件已存在 {edit.path}"
                        )
                        result.success = False
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(edit.new_content or "", encoding="utf-8")
                    result.applied_files.append(edit.path)

                elif edit.operation == EditOperation.MODIFY:
                    if not target.exists():
                        result.failed_files.append(edit.path)
                        result.errors.append(
                            f"MODIFY 失败: 文件不存在 {edit.path}"
                        )
                        result.success = False
                        continue
                    current = target.read_text(encoding="utf-8")
                    if edit.old_content is not None and current != edit.old_content:
                        result.failed_files.append(edit.path)
                        result.errors.append(
                            f"MODIFY 冲突: {edit.path} 的当前内容与预期不一致"
                        )
                        result.success = False
                        continue
                    target.write_text(edit.new_content or "", encoding="utf-8")
                    result.applied_files.append(edit.path)

                elif edit.operation == EditOperation.DELETE:
                    if not target.exists():
                        result.failed_files.append(edit.path)
                        result.errors.append(
                            f"DELETE 失败: 文件不存在 {edit.path}"
                        )
                        result.success = False
                        continue
                    target.unlink()
                    result.applied_files.append(edit.path)

                elif edit.operation == EditOperation.RENAME:
                    old_target = root / edit.old_path if edit.old_path else None
                    if old_target is None or not old_target.exists():
                        result.failed_files.append(edit.path)
                        result.errors.append(
                            f"RENAME 失败: 源文件不存在 {edit.old_path}"
                        )
                        result.success = False
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(str(old_target), str(target))
                    # 如果有新内容，写入
                    if edit.new_content is not None:
                        target.write_text(edit.new_content, encoding="utf-8")
                    result.applied_files.append(edit.path)

            except OSError as e:
                result.failed_files.append(edit.path)
                result.errors.append(f"{edit.operation.value} 异常: {edit.path}: {e}")
                result.success = False

        return result

    def is_empty(self) -> bool:
        """是否没有任何文件变更."""
        return len(self.edits) == 0


def generate_unified_diff(
    path: str,
    operation: EditOperation,
    old_content: str | None,
    new_content: str | None,
    old_path: str | None = None,
) -> str:
    """根据操作类型生成标准 unified diff."""
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = (new_content or "").splitlines(keepends=True)

    if operation == EditOperation.CREATE:
        from_file = "/dev/null"
        to_file = f"b/{path}"
    elif operation == EditOperation.DELETE:
        from_file = f"a/{path}"
        to_file = "/dev/null"
    elif operation == EditOperation.RENAME:
        from_file = f"a/{old_path or path}"
        to_file = f"b/{path}"
    else:  # MODIFY
        from_file = f"a/{path}"
        to_file = f"b/{path}"

    diff_text = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_file,
            tofile=to_file,
        )
    )

    # RENAME 时即使内容相同也要输出头部（类似 git diff --stat）
    if not diff_text and operation == EditOperation.RENAME:
        diff_text = f"--- {from_file}\n+++ {to_file}\n"

    return diff_text
