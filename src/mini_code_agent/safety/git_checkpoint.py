"""Git Checkpoint — Agent 自动创建 checkpoint 并支持回滚."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from ..tools.git import _run_git

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "[agent-checkpoint]"


@dataclass
class CheckpointInfo:
    """一条 checkpoint 的信息."""

    commit_hash: str
    message: str  # 不含 prefix
    timestamp: str


class GitCheckpoint:
    """管理 Agent 的 git checkpoint.

    在任务开始/结束时自动创建 checkpoint commit，
    支持回滚到指定 checkpoint。
    """

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd
        self._enabled: bool | None = None  # lazy 检查

    async def is_git_repo(self) -> bool:
        """检查当前目录是否在 git 仓库中."""
        if self._enabled is not None:
            return self._enabled
        code, _ = await _run_git(
            "rev-parse", "--is-inside-work-tree", cwd=self.cwd
        )
        self._enabled = code == 0
        return self._enabled

    async def create_checkpoint(self, message: str) -> str | None:
        """创建一个 checkpoint commit.

        会 stage 所有改动（包括 untracked），然后提交。
        如果没有任何改动，跳过提交。

        Returns:
            commit hash，或 None（无改动 / 非 git 仓库）。
        """
        if not await self.is_git_repo():
            logger.warning("不在 git 仓库中，跳过 checkpoint")
            return None

        # stage 所有改动（包括 untracked）
        await _run_git("add", "-A", cwd=self.cwd)

        # 检查是否有东西要提交
        code, status = await _run_git("diff", "--cached", "--quiet", cwd=self.cwd)
        if code == 0:
            # 没有 staged 改动
            logger.debug("没有改动，跳过 checkpoint: %s", message)
            return None

        full_message = f"{CHECKPOINT_PREFIX} {message}"
        code, out = await _run_git(
            "commit", "-m", full_message, "--allow-empty", cwd=self.cwd
        )
        if code != 0:
            logger.error("创建 checkpoint 失败: %s", out)
            return None

        # 获取 commit hash
        code, hash_out = await _run_git("rev-parse", "HEAD", cwd=self.cwd)
        commit_hash = hash_out.strip()
        logger.info("创建 checkpoint: %s (%s)", message, commit_hash[:8])
        return commit_hash

    async def rollback_to(self, commit_hash: str) -> bool:
        """回滚到指定 commit（git reset --hard）.

        Returns:
            是否成功。
        """
        if not await self.is_git_repo():
            logger.warning("不在 git 仓库中，无法回滚")
            return False

        code, out = await _run_git("reset", "--hard", commit_hash, cwd=self.cwd)
        if code != 0:
            logger.error("回滚失败: %s", out)
            return False

        logger.info("已回滚到 %s", commit_hash[:8])
        return True

    async def rollback_last(self) -> bool:
        """回滚到最近一个 checkpoint.

        找到最近的 [agent-checkpoint] commit，
        然后 reset --hard 到它的父提交。

        Returns:
            是否成功。
        """
        checkpoints = await self.list_checkpoints()
        if not checkpoints:
            logger.warning("没有找到 checkpoint")
            return False

        latest = checkpoints[0]  # 最近的
        # reset 到该 checkpoint 的父提交（即恢复到 checkpoint 创建前的状态）
        code, out = await _run_git(
            "reset", "--hard", f"{latest.commit_hash}~1", cwd=self.cwd
        )
        if code != 0:
            logger.error("回滚失败: %s", out)
            return False

        logger.info("已回滚到 checkpoint '%s' 之前的状态", latest.message)
        return True

    async def list_checkpoints(self) -> list[CheckpointInfo]:
        """列出所有 agent checkpoint（按时间倒序）.

        Returns:
            CheckpointInfo 列表，最近的在前。
        """
        if not await self.is_git_repo():
            return []

        code, out = await _run_git(
            "log", "--all", "--fixed-strings",
            f"--grep={CHECKPOINT_PREFIX}",
            "--format=%H|%s|%ci",
            cwd=self.cwd,
        )
        if code != 0 or not out.strip():
            return []

        result: list[CheckpointInfo] = []
        for line in out.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            commit_hash, subject, timestamp = parts
            # 去掉 prefix
            msg = subject.replace(CHECKPOINT_PREFIX, "").strip()
            result.append(CheckpointInfo(
                commit_hash=commit_hash.strip(),
                message=msg,
                timestamp=timestamp.strip(),
            ))

        return result

    async def cleanup_checkpoints(self, keep_last_n: int = 5) -> int:
        """清理旧的 checkpoint commit.

        保留最近 keep_last_n 个 checkpoint，将更早的 checkpoint 标记清除。
        注意：使用 soft reset + recommit 而非 interactive rebase（避免交互式操作）。

        Returns:
            清理的 checkpoint 数量。
        """
        checkpoints = await self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return 0

        # 需要清理的 = 超过 keep_last_n 的部分
        to_clean = checkpoints[keep_last_n:]
        cleaned = 0

        for cp in to_clean:
            # 使用 git notes 标记已清理（不改动 commit 历史）
            code, _ = await _run_git(
                "notes", "add", "-m", "[cleaned]", cp.commit_hash,
                cwd=self.cwd,
            )
            if code == 0:
                cleaned += 1

        if cleaned:
            logger.info("已标记清理 %d 个旧 checkpoint", cleaned)
        return cleaned
