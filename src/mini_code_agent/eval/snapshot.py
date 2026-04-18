"""Workspace 文件快照与 diff.

用途：eval runner 在 Agent 跑之前先 `capture(workspace)` 一份指纹，
跑完后用 `diff(workspace, before)` 得到被改动的文件列表。这比 Agent
自报的 `_files_changed` 可靠，因为 Bash 用 `sed -i`/`echo >` 改的文件
Agent 自己追踪不到（见 DESIGN.md §9.4）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


# 快照遍历要忽略的路径片段（与 task_hash 共享语义，但多忽略 .git 这种只在运行期出现的目录）
# - `.agent-backups`：WriteFileTool 写文件前会自动备份，不是 Agent 的"产出"
# - `.pytest_cache`：Agent 在 workspace 里跑 pytest 时副产物，不是修改意图
# 这两个不过滤会把 edit_precision 拉低，污染 files_changed_actual（DESIGN §9.8）。
_IGNORE_NAMES: frozenset[str] = frozenset(
    {"__pycache__", ".DS_Store", ".git", ".agent-backups", ".pytest_cache"}
)
_IGNORE_SUFFIXES: frozenset[str] = frozenset({".pyc"})


@dataclass(frozen=True)
class FileFingerprint:
    """单个文件的指纹：大小 + mtime（ns）+ sha256 内容摘要."""

    size: int
    mtime_ns: int
    sha256: str


@dataclass
class SnapshotDiff:
    """两次 snapshot 之间的差异."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[str]:
        """全部被 touched 的相对路径（added ∪ modified ∪ removed），稳定排序."""
        return sorted(set(self.added) | set(self.modified) | set(self.removed))

    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.removed)


def _should_ignore(rel: Path) -> bool:
    if any(part in _IGNORE_NAMES for part in rel.parts):
        return True
    if rel.suffix in _IGNORE_SUFFIXES:
        return True
    return False


def capture(workspace: Path) -> dict[str, FileFingerprint]:
    """递归扫描 workspace 下所有文件，返回 {相对路径: FileFingerprint}.

    忽略 __pycache__/、*.pyc、.DS_Store、.git/。workspace 不存在会抛 FileNotFoundError。
    """
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace 目录不存在: {workspace}")

    result: dict[str, FileFingerprint] = {}
    for p in workspace.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(workspace)
        if _should_ignore(rel):
            continue
        st = p.stat()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        result[str(rel)] = FileFingerprint(
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            sha256=digest,
        )
    return result


def diff(
    workspace: Path,
    before: dict[str, FileFingerprint],
) -> SnapshotDiff:
    """对比 before 快照与 workspace 当前状态，返回差异.

    only-in-before → removed
    only-in-after  → added
    both-but-sha256-不同 → modified
    """
    after = capture(workspace)
    before_keys = set(before.keys())
    after_keys = set(after.keys())

    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    modified = sorted(
        k for k in (before_keys & after_keys)
        if before[k].sha256 != after[k].sha256
    )

    return SnapshotDiff(added=added, modified=modified, removed=removed)
