"""越界检查 — 检查 Worker 是否只修改了允许的路径."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass
class ScopeCheck:
    """越界检查结果."""

    allowed_paths: list[str]  # 任务允许修改的路径（含通配符）
    touched_paths: list[str]  # 实际修改的路径
    out_of_scope_paths: list[str]  # 越界的路径（touched - allowed）
    is_clean: bool  # out_of_scope_paths 为空


class ScopeChecker:
    """路径越界检查器，支持 glob 模式."""

    @staticmethod
    def check(allowed: list[str], touched: list[str]) -> ScopeCheck:
        """检查 touched 路径是否都在 allowed 范围内.

        支持 glob 模式：
        - src/auth/** 匹配 src/auth/ 下任何文件（任意深度）
        - *.py 匹配当前目录下所有 .py 文件
        - src/*.py 匹配 src/ 下的 .py 文件
        - 保留文件（如 README.md）应该在 allowed 中显式列出
        """
        out_of_scope: list[str] = []

        for path in touched:
            if not ScopeChecker._is_allowed(path, allowed):
                out_of_scope.append(path)

        return ScopeCheck(
            allowed_paths=allowed,
            touched_paths=touched,
            out_of_scope_paths=out_of_scope,
            is_clean=len(out_of_scope) == 0,
        )

    @staticmethod
    def _is_allowed(path: str, allowed: list[str]) -> bool:
        """检查单个路径是否被允许."""
        for pattern in allowed:
            # 处理 ** 通配符：src/auth/** 匹配 src/auth/ 下任意深度的文件
            if pattern.endswith("/**"):
                prefix = pattern[:-3]  # 去掉 /**
                if path == prefix or path.startswith(prefix + "/"):
                    return True
            # 精确匹配或 fnmatch 模式匹配
            if path == pattern or fnmatch(path, pattern):
                return True
        return False
