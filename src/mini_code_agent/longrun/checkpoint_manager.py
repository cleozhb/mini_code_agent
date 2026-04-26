"""CheckpointManager — Checkpoint 的创建、加载、清理与自动触发策略."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..core.task_graph import TaskGraph
from ..safety.git_checkpoint import GitCheckpoint
from ..tools.git import _run_git
from .config import LongRunConfig
from .ledger_manager import TaskLedgerManager
from .session_state import CheckpointMeta, CheckpointTrigger, SessionState
from .task_ledger import TaskLedger

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Checkpoint 操作错误."""


class CorruptedCheckpointError(CheckpointError):
    """Checkpoint 数据不一致（ledger_hash / git_hash 验证失败）."""


def _compute_sha256(path: str) -> str:
    """计算文件的 SHA256 摘要."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: str, content: str) -> None:
    """原子写入文件：先写 .tmp 再 rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(p) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, str(p))


def _summarize_messages(messages: list[dict], max_length: int = 500) -> str:
    """从消息列表中生成简要摘要."""
    if not messages:
        return "(无最近消息)"
    lines: list[str] = []
    for msg in messages[-5:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            preview = content[:100].replace("\n", " ")
        else:
            preview = str(content)[:100]
        lines.append(f"[{role}] {preview}")
    summary = "\n".join(lines)
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return summary


class CheckpointManager:
    """管理 Checkpoint 的创建、加载和清理.

    文件布局:
        .agent/checkpoints/
            {task_id}/
                {checkpoint_id}.json     # SessionState 序列化
                index.json               # 该任务所有 checkpoint 的索引（按时间排序）
    """

    def __init__(
        self,
        checkpoint_dir: str = ".agent/checkpoints/",
        ledger_manager: TaskLedgerManager | None = None,
        git_checkpoint: GitCheckpoint | None = None,
        cwd: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_manager = ledger_manager
        self.git_checkpoint = git_checkpoint
        self.cwd = cwd

    async def save_checkpoint(
        self,
        ledger: TaskLedger,
        task_graph: TaskGraph,
        trigger: CheckpointTrigger,
        config: LongRunConfig,
        current_task_id: str | None,
        recent_messages: list[dict],
    ) -> SessionState:
        """创建一个完整的 checkpoint.

        执行步骤（顺序敏感）：
        1. 前置检查：工作区是否干净
        2. 创建 git checkpoint
        3. 确保 ledger 已保存
        4. 构造 SessionState
        5. 原子写入
        6. 清理旧 checkpoint
        """
        task_id = ledger.task_id
        had_uncommitted = False

        # 步骤 1 — 前置检查：检查是否有未提交的修改
        code, status_output = await _run_git("status", "--porcelain", cwd=self.cwd)
        if code == 0 and status_output.strip():
            # 有未提交的修改 — 这不应该发生
            # 选方案一：自动 commit 这些修改，标记 uncommitted_changes=True
            logger.warning(
                "Checkpoint 前发现未提交修改，自动 commit: %s",
                status_output.strip()[:200],
            )
            had_uncommitted = True
            await _run_git("add", "-A", cwd=self.cwd)
            await _run_git(
                "commit", "-m",
                f"[agent-checkpoint] auto-commit dirty workspace: "
                f"task={task_id} trigger={trigger.value}",
                cwd=self.cwd,
            )

        # 步骤 2 — 创建 git checkpoint
        git_hash = ""
        if self.git_checkpoint is not None:
            result = await self.git_checkpoint.create_checkpoint(
                f"task={task_id} trigger={trigger.value}"
            )
            if result:
                git_hash = result
            else:
                # create_checkpoint 返回 None 说明没有新改动
                # 直接取当前 HEAD 作为 checkpoint hash
                code, head = await _run_git("rev-parse", "HEAD", cwd=self.cwd)
                if code == 0:
                    git_hash = head.strip()
        else:
            # 无 git checkpoint manager，直接取 HEAD
            code, head = await _run_git("rev-parse", "HEAD", cwd=self.cwd)
            if code == 0:
                git_hash = head.strip()

        # 步骤 3 — 确保 ledger 已保存
        if self.ledger_manager is not None:
            self.ledger_manager.save(ledger)
        ledger_path = str(
            self.ledger_manager.storage_dir / f"{task_id}.json"
            if self.ledger_manager
            else f".agent/ledger/{task_id}.json"
        )
        ledger_hash = _compute_sha256(ledger_path) if Path(ledger_path).exists() else ""

        # 获取当前分支
        code, branch = await _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.cwd)
        git_branch = branch.strip() if code == 0 else "unknown"

        # 步骤 4 — 构造 SessionState
        checkpoint_id = str(uuid4())
        now = datetime.now(UTC)

        # 消息序列化：只保留最近 20 条的 role/content
        recent_full = None
        if recent_messages:
            recent_full = []
            for msg in recent_messages[-20:]:
                if isinstance(msg, dict):
                    recent_full.append(msg)
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    # Message 对象
                    entry: dict = {"role": msg.role.value if hasattr(msg.role, "value") else str(msg.role)}
                    if isinstance(msg.content, str):
                        entry["content"] = msg.content[:500]
                    else:
                        entry["content"] = str(msg.content)[:500]
                    recent_full.append(entry)

        state = SessionState(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            created_at=now,
            trigger=trigger,
            ledger_path=ledger_path,
            ledger_hash=ledger_hash,
            task_graph_json=task_graph.to_json(),
            current_task_id=current_task_id,
            git_checkpoint_hash=git_hash,
            git_branch=git_branch,
            uncommitted_changes=had_uncommitted,
            config_snapshot=config.to_dict(),
            step_number=ledger.total_steps,
            recent_messages_summary=_summarize_messages(
                recent_full or []
            ),
            recent_messages_full=recent_full,
        )

        # 步骤 5 — 原子写入
        state_path = str(self.checkpoint_dir / task_id / f"{checkpoint_id}.json")
        _write_atomic(state_path, state.to_json())
        self._update_index(task_id, state, ledger)

        # 步骤 6 — 清理旧 checkpoint
        self.cleanup_old_checkpoints(task_id, keep_last_n=config.max_checkpoints)

        logger.info(
            "Checkpoint 已创建: %s (trigger=%s, step=%d, git=%s)",
            checkpoint_id[:8], trigger.value, state.step_number,
            git_hash[:8] if git_hash else "none",
        )

        return state

    def list_checkpoints(self, task_id: str) -> list[CheckpointMeta]:
        """读取 index.json，返回按时间倒序的 checkpoint 列表."""
        index_path = self.checkpoint_dir / task_id / "index.json"
        if not index_path.exists():
            return []

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 checkpoint index 失败: %s", e)
            return []

        metas = [CheckpointMeta.from_dict(item) for item in data.get("checkpoints", [])]
        # 按时间倒序
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def load_checkpoint(self, task_id: str, checkpoint_id: str) -> SessionState:
        """从磁盘读取 SessionState 并验证一致性.

        验证：
        - ledger_hash 是否和当前 ledger 文件匹配
        - git_checkpoint_hash 是否仍存在于 git 历史

        任何验证失败都抛出 CorruptedCheckpointError。
        """
        state_path = self.checkpoint_dir / task_id / f"{checkpoint_id}.json"
        if not state_path.exists():
            raise CheckpointError(f"Checkpoint 文件不存在: {state_path}")

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise CheckpointError(f"读取 checkpoint 失败: {e}") from e

        state = SessionState.from_dict(data)
        return state

    async def validate_checkpoint(self, state: SessionState) -> list[str]:
        """验证 checkpoint 的一致性，返回 warnings 列表.

        如果 git hash 不存在于历史中，抛出 CorruptedCheckpointError。
        """
        warnings: list[str] = []

        # 验证 ledger hash
        if state.ledger_path and Path(state.ledger_path).exists():
            current_hash = _compute_sha256(state.ledger_path)
            if current_hash != state.ledger_hash:
                warnings.append(
                    "Ledger 在 checkpoint 创建后被修改过。"
                    "恢复时会使用当前的 Ledger，请检查是否符合预期。"
                )
        elif state.ledger_path:
            warnings.append(f"Ledger 文件不存在: {state.ledger_path}")

        # 验证 git hash 存在
        if state.git_checkpoint_hash:
            code, _ = await _run_git(
                "cat-file", "-t", state.git_checkpoint_hash, cwd=self.cwd
            )
            if code != 0:
                raise CorruptedCheckpointError(
                    f"Git commit {state.git_checkpoint_hash[:8]} 不存在于历史中。"
                    f"该 checkpoint 已损坏，无法恢复。"
                )

        return warnings

    def find_latest(self, task_id: str) -> SessionState | None:
        """查找指定任务最近的 checkpoint."""
        metas = self.list_checkpoints(task_id)
        if not metas:
            return None
        latest = metas[0]
        return self.load_checkpoint(task_id, latest.id)

    def cleanup_old_checkpoints(self, task_id: str, keep_last_n: int = 5) -> int:
        """按时间排序，删除超过 keep_last_n 的旧 checkpoint JSON 文件.

        注意：不删除对应的 git commit（那是 git gc 的事）。
        """
        metas = self.list_checkpoints(task_id)
        if len(metas) <= keep_last_n:
            return 0

        # metas 已按时间倒序排列，保留前 keep_last_n 个
        to_delete = metas[keep_last_n:]
        deleted = 0

        for meta in to_delete:
            state_path = self.checkpoint_dir / task_id / f"{meta.id}.json"
            try:
                if state_path.exists():
                    state_path.unlink()
                    deleted += 1
            except OSError as e:
                logger.warning("删除旧 checkpoint 失败: %s", e)

        # 更新 index
        if deleted > 0:
            remaining = metas[:keep_last_n]
            self._write_index(task_id, remaining)
            logger.info("清理了 %d 个旧 checkpoint（任务 %s）", deleted, task_id[:8])

        return deleted

    def auto_checkpoint_policy(
        self,
        ledger: TaskLedger,
        last_checkpoint: SessionState | None,
        config: LongRunConfig,
    ) -> CheckpointTrigger | None:
        """判断是否需要自动创建 checkpoint.

        返回 None 表示不需要，返回 trigger 表示需要。
        策略：
        - 距上次 checkpoint 的 token 增量 >= config.checkpoint_interval_tokens
          → TOKEN_THRESHOLD
        - 子任务刚完成且 config.checkpoint_on_subtask_complete
          → SUBTASK_COMPLETE
        - 阶段切换 → PHASE_TRANSITION
        """
        if last_checkpoint is None:
            # 还没有 checkpoint，如果已经消耗了一定 token 就创建
            if ledger.total_tokens_used >= config.checkpoint_interval_tokens:
                return CheckpointTrigger.TOKEN_THRESHOLD
            return None

        # Token 增量检查
        tokens_since = ledger.total_tokens_used - last_checkpoint.step_number
        # 用 step_number 来近似 token 增量（实际应该用 token 计数，
        # 但 step_number 存在 checkpoint 里，更可靠）
        # 重新计算：直接看 ledger 的总 token 与 checkpoint 的 config 里的记录
        last_tokens = last_checkpoint.config_snapshot.get("_tokens_at_checkpoint", 0)
        token_delta = ledger.total_tokens_used - last_tokens
        if token_delta >= config.checkpoint_interval_tokens:
            return CheckpointTrigger.TOKEN_THRESHOLD

        return None

    def _update_index(
        self,
        task_id: str,
        state: SessionState,
        ledger: TaskLedger,
    ) -> None:
        """更新任务的 checkpoint index.json."""
        index_path = self.checkpoint_dir / task_id / "index.json"

        # 读取现有 index
        existing_metas: list[dict] = []
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                existing_metas = data.get("checkpoints", [])
            except (json.JSONDecodeError, OSError):
                existing_metas = []

        # 追加新条目
        meta = CheckpointMeta(
            id=state.checkpoint_id,
            created_at=state.created_at,
            trigger=state.trigger,
            step_number=state.step_number,
            token_count=ledger.total_tokens_used,
            git_hash=state.git_checkpoint_hash,
            current_task_id=state.current_task_id,
        )
        existing_metas.append(meta.to_dict())

        # 原子写入 index
        index_data = {"task_id": task_id, "checkpoints": existing_metas}
        _write_atomic(
            str(index_path),
            json.dumps(index_data, ensure_ascii=False, indent=2),
        )

    def _write_index(self, task_id: str, metas: list[CheckpointMeta]) -> None:
        """用 metas 列表覆写 index.json."""
        index_path = self.checkpoint_dir / task_id / "index.json"
        index_data = {
            "task_id": task_id,
            "checkpoints": [m.to_dict() for m in metas],
        }
        _write_atomic(
            str(index_path),
            json.dumps(index_data, ensure_ascii=False, indent=2),
        )
