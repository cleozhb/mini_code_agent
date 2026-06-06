"""Path filters for agent-generated noise and internal state."""

from __future__ import annotations

from pathlib import PurePosixPath

_IGNORED_NAMES = {
    ".agent",
    ".agent-backups",
    ".pytest_cache",
    "__pycache__",
}

_IGNORED_EXACT_PATHS = {
    ".agent_history",
}

_IGNORED_SUFFIXES = {
    ".pyc",
}


def is_agent_internal_path(path: str) -> bool:
    """Return True for agent/runtime artifacts that should not enter patches."""
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False

    p = PurePosixPath(normalized)
    if normalized in _IGNORED_EXACT_PATHS:
        return True
    if p.suffix in _IGNORED_SUFFIXES:
        return True
    return any(part in _IGNORED_NAMES for part in p.parts)


def path_from_git_status_entry(entry: str) -> str:
    """Extract the target path from a `git status --porcelain` line."""
    raw = entry[3:] if len(entry) > 3 else entry
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip()
