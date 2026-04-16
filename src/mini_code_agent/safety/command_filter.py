"""命令过滤器 — 对 Shell 命令进行安全分级."""

from __future__ import annotations

import re
import shlex
from enum import Enum


class SafetyLevel(str, Enum):
    """命令安全等级."""

    SAFE = "safe"                # 白名单命令，自动放行
    NEEDS_CONFIRM = "needs_confirm"  # 需要用户确认
    BLOCKED = "blocked"          # 黑名单命令，直接拒绝


# ---------------------------------------------------------------------------
# 白名单：这些命令或前缀自动放行
# ---------------------------------------------------------------------------

WHITELIST_COMMANDS: list[str] = [
    "ls",
    "cat",
    "head",
    "tail",
    "find",
    "grep",
    "wc",
    "pwd",
    "echo",
    "python --version",
    "python3 --version",
    "node --version",
    "git status",
    "git log",
    "git diff",
]

# ---------------------------------------------------------------------------
# 黑名单：匹配到即拒绝
# ---------------------------------------------------------------------------

BLACKLIST_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",          # rm -rf /
    r"rm\s+-rf\s+~",          # rm -rf ~
    r"\bsudo\b",              # sudo
    r"chmod\s+777",           # chmod 777
    r">\s*/dev/",             # > /dev/sda etc
    r"curl\s+.*\|\s*bash",   # curl | bash
    r"curl\s+.*\|\s*sh",     # curl | sh
    r"wget\s+.*\|\s*sh",     # wget | sh
    r"wget\s+.*\|\s*bash",   # wget | bash
    r":\(\)\s*\{",           # fork bomb
    r"\bmkfs\b",             # mkfs
    r"\bdd\s+if=",           # dd if=
]

# ---------------------------------------------------------------------------
# 敏感路径：访问这些路径的命令需要额外确认
# ---------------------------------------------------------------------------

SENSITIVE_PATHS: list[str] = [
    "~/.ssh",
    "~/.aws",
    "~/.config",
    "/etc/",
    ".env",
]


class CommandFilter:
    """Shell 命令安全过滤器.

    使用白名单/黑名单/敏感路径三层检查，对命令进行安全分级。
    """

    def __init__(
        self,
        whitelist: list[str] | None = None,
        blacklist_patterns: list[str] | None = None,
        sensitive_paths: list[str] | None = None,
    ) -> None:
        self.whitelist = whitelist or WHITELIST_COMMANDS
        self.blacklist_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (blacklist_patterns or BLACKLIST_PATTERNS)
        ]
        self.sensitive_paths = sensitive_paths or SENSITIVE_PATHS

    def is_safe(self, command: str) -> SafetyLevel:
        """判断命令的安全等级.

        检查顺序：
        1. 黑名单 → BLOCKED
        2. 敏感路径 → NEEDS_CONFIRM
        3. 白名单 → SAFE
        4. 其他 → NEEDS_CONFIRM
        """
        cmd = command.strip()

        # 1) 黑名单检查（最高优先级）
        for pattern in self.blacklist_patterns:
            if pattern.search(cmd):
                return SafetyLevel.BLOCKED

        # 2) 敏感路径检查
        if self._touches_sensitive_path(cmd):
            return SafetyLevel.NEEDS_CONFIRM

        # 3) 白名单检查
        if self._is_whitelisted(cmd):
            return SafetyLevel.SAFE

        # 4) 默认需要确认
        return SafetyLevel.NEEDS_CONFIRM

    def get_block_reason(self, command: str) -> str | None:
        """如果命令被拦截，返回拦截原因."""
        cmd = command.strip()
        for pattern in self.blacklist_patterns:
            if pattern.search(cmd):
                return f"命令匹配危险模式: {pattern.pattern}"
        return None

    def _is_whitelisted(self, command: str) -> bool:
        """检查命令是否在白名单中."""
        # 处理管道：只看第一个命令
        first_cmd = command.split("|")[0].strip()

        for wl in self.whitelist:
            # 精确匹配整个命令（如 "python --version"）
            if first_cmd == wl:
                return True
            # 前缀匹配：
            #   单词白名单（如 "ls"）→ 匹配 "ls -la"
            #   多词白名单（如 "git log"）→ 匹配 "git log --oneline -10"
            if first_cmd.startswith(wl) and (
                len(first_cmd) == len(wl) or first_cmd[len(wl)] == " "
            ):
                return True
        return False

    def _touches_sensitive_path(self, command: str) -> bool:
        """检查命令是否涉及敏感路径."""
        for path in self.sensitive_paths:
            if path in command:
                return True
        return False

    @staticmethod
    def _command_name(cmd: str) -> str:
        """提取命令名称（第一个 token）."""
        try:
            tokens = shlex.split(cmd)
            return tokens[0] if tokens else ""
        except ValueError:
            # shlex 解析失败时，简单 split
            parts = cmd.split()
            return parts[0] if parts else ""
