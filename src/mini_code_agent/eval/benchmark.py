"""Benchmark 任务与任务集定义.

提供：
- BenchmarkTask: 单个 eval 任务（YAML + workspace/ + validate.py）
- BenchmarkSuite: 任务集合 + 过滤器
- compute_task_hash / compute_suite_hash: 版本化 hash

详见 DESIGN.md §2、§9.1。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

# hash 计算和 suite 遍历都需要忽略的路径片段
_IGNORE_NAMES: frozenset[str] = frozenset({"__pycache__", ".DS_Store"})
_IGNORE_SUFFIXES: frozenset[str] = frozenset({".pyc"})


# ---------------------------------------------------------------------------
# Hash 计算
# ---------------------------------------------------------------------------


def _iter_hashable_files(root: Path) -> list[Path]:
    """遍历 root 下所有需要参与 hash 的文件（忽略 pycache 等），按相对路径排序."""
    files: list[tuple[str, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _IGNORE_NAMES for part in rel.parts):
            continue
        if p.suffix in _IGNORE_SUFFIXES:
            continue
        files.append((str(rel), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def compute_task_hash(task_dir: Path) -> str:
    """对 task 目录内容算 sha256.

    规则：
    - 遍历所有文件（忽略 __pycache__/*.pyc/.DS_Store），按相对路径排序
    - 每个文件先算 sha256(relpath_bytes || b"\\0" || content)
    - 把这些 hex digest 用换行拼起来再 sha256
    """
    task_dir = Path(task_dir)
    h = hashlib.sha256()
    for f in _iter_hashable_files(task_dir):
        rel = str(f.relative_to(task_dir)).encode("utf-8")
        file_digest = hashlib.sha256(
            rel + b"\0" + f.read_bytes()
        ).hexdigest()
        h.update(file_digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def compute_suite_hash(tasks: list["BenchmarkTask"]) -> str:
    """对一组任务算 suite hash：sorted([f"{id}:{task_hash}" for t in tasks]) join 后 sha256."""
    parts = sorted(f"{t.id}:{t.task_hash}" for t in tasks)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BenchmarkTask
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS = (
    "id",
    "level",
    "description",
    "max_steps",
    "max_tokens",
    "max_wall_time_seconds",
)


@dataclass(frozen=True)
class BenchmarkTask:
    """单个 benchmark 任务的不可变描述."""

    id: str
    level: int
    description: str
    workspace_dir: Path
    validate_script: Path
    expected_files: tuple[str, ...]
    max_steps: int
    max_tokens: int
    max_wall_time_seconds: int
    tags: tuple[str, ...]
    task_hash: str

    @classmethod
    def load(cls, task_dir: Path) -> "BenchmarkTask":
        """从 task_dir 加载：读 task.yaml + 定位 workspace/ + validate.py + 算 hash.

        期望目录结构：
            task_dir/
              task.yaml
              workspace/...
              validate.py
        """
        task_dir = Path(task_dir).resolve()
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"缺少 task.yaml: {yaml_path}")

        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"task.yaml 解析失败: {yaml_path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"task.yaml 顶层必须是映射: {yaml_path}")

        missing = [k for k in _REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(
                f"task.yaml 缺少必需字段 {missing}: {yaml_path}"
            )

        workspace_dir = task_dir / "workspace"
        validate_script = task_dir / "validate.py"
        if not workspace_dir.is_dir():
            raise FileNotFoundError(f"缺少 workspace 目录: {workspace_dir}")
        if not validate_script.is_file():
            raise FileNotFoundError(f"缺少 validate.py: {validate_script}")

        return cls(
            id=str(data["id"]),
            level=int(data["level"]),
            description=str(data["description"]).strip(),
            workspace_dir=workspace_dir,
            validate_script=validate_script,
            expected_files=tuple(str(x) for x in data.get("expected_files", [])),
            max_steps=int(data["max_steps"]),
            max_tokens=int(data["max_tokens"]),
            max_wall_time_seconds=int(data["max_wall_time_seconds"]),
            tags=tuple(str(x) for x in data.get("tags", [])),
            task_hash=compute_task_hash(task_dir),
        )


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSuite:
    """多个 BenchmarkTask 的集合，附带整体 suite_hash."""

    tasks: list[BenchmarkTask]
    suite_hash: str

    @classmethod
    def load_from_dir(cls, tasks_dir: Path) -> "BenchmarkSuite":
        """遍历 tasks_dir 下每个子目录，凡是含 task.yaml 的都算一个任务."""
        tasks_dir = Path(tasks_dir)
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"tasks 目录不存在: {tasks_dir}")

        task_dirs = sorted(
            p for p in tasks_dir.iterdir()
            if p.is_dir() and (p / "task.yaml").is_file()
        )
        tasks = [BenchmarkTask.load(p) for p in task_dirs]
        return cls(tasks=tasks, suite_hash=compute_suite_hash(tasks))

    def filter_by_level(self, level: int) -> "BenchmarkSuite":
        filtered = [t for t in self.tasks if t.level == level]
        return BenchmarkSuite(tasks=filtered, suite_hash=compute_suite_hash(filtered))

    def filter_by_tag(self, tag: str) -> "BenchmarkSuite":
        filtered = [t for t in self.tasks if tag in t.tags]
        return BenchmarkSuite(tasks=filtered, suite_hash=compute_suite_hash(filtered))

    def get(self, task_id: str) -> BenchmarkTask | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def __iter__(self) -> Iterator[BenchmarkTask]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)
