"""Repo Map 生成 — 为整个项目生成文件级别的"地图"."""

from __future__ import annotations

from pathlib import Path

from .project_analyzer import _should_ignore, summarize_file

# 只对这些后缀的文件生成摘要，其他文件只列路径
_CODE_SUFFIXES: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php",
}

# 配置文件等只列路径不做摘要
_CONFIG_SUFFIXES: set[str] = {
    ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
    ".xml", ".md", ".rst", ".txt",
    ".html", ".css", ".sql", ".sh",
    ".dockerfile", ".env",
}


def build_repo_map(
    path: str | Path,
    *,
    max_files: int = 500,
    include_signatures: bool = True,
) -> str:
    """为整个项目生成一张"地图"：每个文件一行，包含文件路径 + 简要内容摘要.

    输出示例：
        src/mini_code_agent/core/agent.py — Agent class: run(), reset()
        src/mini_code_agent/tools/shell.py — BashTool: execute shell commands
        src/mini_code_agent/llm/base.py — LLMClient ABC, Message, ToolCall dataclasses

    Args:
        path: 项目根目录
        max_files: 最大文件数量，超过则只列路径不做签名摘要
        include_signatures: 是否包含函数/类签名摘要

    Returns:
        整个 repo map 的字符串
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return f"[不是目录: {root}]"

    # 收集所有文件
    all_files = _collect_files(root)

    # 如果文件数超过 max_files，降级为只列路径
    force_path_only = len(all_files) > max_files

    lines: list[str] = []
    for file_path in all_files:
        rel_path = file_path.relative_to(root)

        if force_path_only or not include_signatures:
            lines.append(str(rel_path))
            continue

        suffix = file_path.suffix
        if suffix in _CODE_SUFFIXES:
            # 生成摘要
            summary = summarize_file(file_path)
            # summarize_file 返回 "绝对路径 — 摘要"，我们换成相对路径
            if " — " in summary:
                _, desc = summary.split(" — ", 1)
                lines.append(f"{rel_path} — {desc}")
            else:
                lines.append(str(rel_path))
        else:
            lines.append(str(rel_path))

    return "\n".join(lines)


def build_repo_map_paths_only(path: str | Path) -> str:
    """只返回文件路径列表，不做签名摘要（降级模式）."""
    return build_repo_map(path, include_signatures=False)


def _collect_files(root: Path) -> list[Path]:
    """递归收集所有应该纳入 repo map 的文件，按路径排序."""
    files: list[Path] = []
    _walk_collect(root, files)
    files.sort(key=lambda p: str(p))
    return files


def _walk_collect(directory: Path, files: list[Path]) -> None:
    """递归遍历目录，收集文件."""
    try:
        entries = sorted(directory.iterdir(), key=lambda e: e.name)
    except PermissionError:
        return

    for entry in entries:
        if _should_ignore(entry):
            continue

        if entry.is_dir():
            _walk_collect(entry, files)
        elif entry.is_file():
            # 跳过隐藏文件（但保留 .env.example 等配置文件）
            if entry.name.startswith(".") and entry.name not in {
                ".env.example", ".gitignore", ".dockerignore",
                ".cursorrules",
            }:
                continue
            # 跳过二进制文件和锁文件
            if entry.suffix in {".lock", ".png", ".jpg", ".jpeg", ".gif",
                                ".ico", ".woff", ".woff2", ".ttf", ".eot",
                                ".zip", ".tar", ".gz", ".bz2", ".7z",
                                ".exe", ".dll", ".so", ".dylib",
                                ".pyc", ".pyo", ".class", ".o"}:
                continue
            files.append(entry)
