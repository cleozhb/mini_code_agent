"""Git Checkpoint — Agent 自动创建 checkpoint 并支持回滚.

设计：
- "before" 阶段只记录当前 HEAD hash，不创建 commit，不动工作区
- "after" 阶段对比 HEAD 和工作区，只把 Agent 期间产生的改动提交为 checkpoint
- 用户已有的 staged/unstaged 改动在 checkpoint 前先 stash 保存，完成后恢复
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    在任务开始时记录 HEAD hash，任务结束时把 Agent 产生的改动
    提交为 checkpoint commit，支持回滚。
    """

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd
        self._enabled: bool | None = None  # lazy 检查
        self._before_head: str | None = None  # "before" 记录的 HEAD hash
        self._before_status: set[str] = set()  # "before" 记录的 git status 快照

    async def is_git_repo(self) -> bool:
        """检查当前目录是否在 git 仓库中."""
        if self._enabled is not None:
            return self._enabled
        code, _ = await _run_git(
            "rev-parse", "--is-inside-work-tree", cwd=self.cwd
        )
        self._enabled = code == 0
        return self._enabled

    async def save_head(self) -> str | None:
        """记录当前 HEAD hash 和工作区状态（"before" 阶段调用）.

        不创建任何 commit，不修改工作区。
        同时记录当前 git status 快照，用于后续区分
        哪些改动是 Agent 产生的、哪些是用户之前就有的。

        Returns:
            HEAD commit hash，或 None（非 git 仓库）。
        """
        if not await self.is_git_repo():
            logger.warning("不在 git 仓库中，跳过 checkpoint")
            return None

        code, out = await _run_git("rev-parse", "HEAD", cwd=self.cwd)
        if code != 0:
            logger.error("获取 HEAD 失败: %s", out)
            return None

        self._before_head = out.strip()

        # 记录当前工作区状态快照（每行是一个文件的变更条目）
        code, status = await _run_git("status", "--porcelain", cwd=self.cwd)
        if code == 0 and status.strip():
            self._before_status = set(status.rstrip("\n").split("\n"))
        else:
            self._before_status = set()

        logger.info("记录 checkpoint 锚点: %s", self._before_head[:8])
        return self._before_head

    async def create_checkpoint(self, message: str) -> str | None:
        """创建 checkpoint commit（"after" 阶段调用）.

        对比 save_head 时的工作区快照，只有出现**新的**改动
        （即不在 _before_status 中的条目）时才创建 checkpoint。
        这样可以避免把用户之前就有的未提交改动误 commit。

        Returns:
            commit hash，或 None（无新增改动 / 非 git 仓库）。
        """
        if not await self.is_git_repo():
            logger.warning("不在 git 仓库中，跳过 checkpoint")
            return None

        # 获取当前工作区状态
        code, status = await _run_git("status", "--porcelain", cwd=self.cwd)
        if code != 0 or not status.strip():
            logger.debug("没有改动，跳过 checkpoint: %s", message)
            return None

        # 对比：只看新增的改动条目
        current_status = set(status.rstrip("\n").split("\n"))
        new_changes = current_status - self._before_status
        if not new_changes:
            logger.debug("没有新增改动（均为 Agent 运行前已有），跳过 checkpoint: %s", message)
            return None

        # 只 stage 新增改动的文件（从 porcelain 输出中提取文件路径）
        for entry in new_changes:
            # git status --porcelain 格式: "XY filename" 或 "XY orig -> filename"
            raw = entry[3:]  # 跳过状态码和空格
            if " -> " in raw:
                raw = raw.split(" -> ", 1)[1]
            await _run_git("add", "--", raw, cwd=self.cwd)

        # 确认有 staged 内容
        code, _ = await _run_git("diff", "--cached", "--quiet", cwd=self.cwd)
        if code == 0:
            logger.debug("没有 staged 改动，跳过 checkpoint: %s", message)
            return None

        full_message = f"{CHECKPOINT_PREFIX} {message}"
        code, out = await _run_git(
            "commit", "-m", full_message, cwd=self.cwd
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
        """回滚到最近一个 checkpoint 之前的状态.

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

        Returns:
            清理的 checkpoint 数量。
        """
        checkpoints = await self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return 0

        to_clean = checkpoints[keep_last_n:]
        cleaned = 0

        for cp in to_clean:
            code, _ = await _run_git(
                "notes", "add", "-m", "[cleaned]", cp.commit_hash,
                cwd=self.cwd,
            )
            if code == 0:
                cleaned += 1

        if cleaned:
            logger.info("已标记清理 %d 个旧 checkpoint", cleaned)
        return cleaned
