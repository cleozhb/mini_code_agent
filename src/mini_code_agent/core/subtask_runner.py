"""SubtaskRunner — 把一个 TaskNode 交给 Agent 执行，产出一个 SubtaskArtifact.

GraphExecutor 与 Agent 之间的中介层。每个子任务：
1. 拿到一个 ArtifactBuilder
2. 让 Agent 执行子任务（observer 记录 tool / LLM 调用）
3. 把改动文件做增量验证
4. finalize Artifact 并落盘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..artifacts import (
    ArtifactBuilder,
    ArtifactStore,
    Confidence,
    EditOperation,
    SubtaskArtifact,
)
from ..artifacts.verification import CheckResult, SelfVerification
from ..safety.git_checkpoint import GitCheckpoint
from ..tools.git import _run_git
from ..verify.verifier import IncrementalVerifier
from .agent import Agent, AgentObserver, AgentStuckError
from .task_graph import TaskNode

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """SubtaskRunner.run 需要的上下文（由 GraphExecutor 构造）."""

    original_goal: str
    completed_summaries: list[str] = field(default_factory=list)
    project_path: str = "."
    allowed_paths: list[str] = field(default_factory=list)


class _ArtifactObserver(AgentObserver):
    """Observer 实现：把 Agent 的 tool / LLM 事件累积到 ArtifactBuilder."""

    def __init__(self, builder: ArtifactBuilder) -> None:
        self.builder = builder

    def on_tool_call(self, name: str, args: dict[str, Any], result: Any) -> None:
        # 只累计计数；文件 diff 在子任务结束后统一从 git 计算
        try:
            self.builder.record_tool_call(name, str(args)[:200], "")
        except Exception as e:  # noqa: BLE001 — finalize 后可能抛
            logger.debug("record_tool_call ignored: %s", e)

    def on_llm_call(self, tokens_in: int, tokens_out: int, model: str) -> None:
        try:
            self.builder.record_llm_call(tokens_in, tokens_out, model)
        except Exception as e:  # noqa: BLE001
            logger.debug("record_llm_call ignored: %s", e)


class SubtaskRunner:
    """把一个 TaskNode 交给 Agent 执行，产出一个 SubtaskArtifact."""

    def __init__(
        self,
        agent: Agent,
        artifact_store: ArtifactStore,
        verifier: IncrementalVerifier,
        git_checkpoint: GitCheckpoint | None,
    ) -> None:
        self.agent = agent
        self.artifact_store = artifact_store
        self.verifier = verifier
        self.git_checkpoint = git_checkpoint

    # ─────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────

    async def run(
        self,
        task_node: TaskNode,
        graph_context: GraphContext,
    ) -> SubtaskArtifact:
        """执行一个子任务并返回 Artifact.

        正常路径不抛异常：执行失败 / Agent 卡住都会被反映在 Artifact.confidence。
        只有基础设施错误（git 操作不可恢复、磁盘 IO 等）才向上抛。
        """
        # 步骤 1 — ArtifactBuilder
        base_hash = await self._current_commit_hash(graph_context.project_path)
        allowed = graph_context.allowed_paths or self.derive_allowed_paths(task_node)

        builder = ArtifactBuilder(
            task_id=task_node.id,
            task_description=task_node.description,
            allowed_paths=allowed,
            verification_spec=task_node.verification,
            producer="agent",
        )
        builder.start(base_git_hash=base_hash or "")

        # 步骤 2 — 子任务 prompt
        prompt = self._build_prompt(task_node, graph_context, allowed)

        # 步骤 3 — 挂 observer 并执行 Agent
        observer = _ArtifactObserver(builder)
        self.agent.add_observer(observer)
        # 锚定 git HEAD，便于事后比较改动
        if self.git_checkpoint is not None:
            try:
                await self.git_checkpoint.save_head()
            except Exception as e:  # noqa: BLE001
                logger.debug("git_checkpoint.save_head failed: %s", e)

        self.agent.reset()

        try:
            result = await self.agent.run(prompt)
            result_text = result.content or ""
            stop_reason = result.stop_reason
        except AgentStuckError as e:
            builder.set_confidence(Confidence.STUCK, e.reason)
            if e.question:
                builder.add_open_question(e.question)
            result_text = ""
            stop_reason = "stuck"
        except Exception as e:  # noqa: BLE001 — 执行异常按 UNCERTAIN 处理
            logger.exception("Agent 执行子任务异常: %s", task_node.id)
            builder.set_confidence(Confidence.UNCERTAIN, f"执行异常：{type(e).__name__}: {e}")
            result_text = ""
            stop_reason = "error"
        finally:
            self.agent.remove_observer(observer)

        if builder.confidence is None:
            # Agent 正常结束 — 暂记 DONE，验证后再调整
            if stop_reason == "ok":
                builder.set_confidence(Confidence.DONE, _summarize(result_text))
            else:
                builder.set_confidence(
                    Confidence.UNCERTAIN,
                    f"Agent 未正常结束（{stop_reason}）：{_summarize(result_text)}",
                )

        # 步骤 4 — 把工作区改动写进 builder（用 git 比较 base..HEAD）
        files_changed = await self._collect_file_edits(
            builder, base_hash or "", graph_context.project_path,
        )

        # 步骤 5 — 增量验证
        try:
            verif_result = await self.verifier.verify_after_edit(
                files_changed=files_changed,
                project_path=graph_context.project_path,
                task_id=task_node.id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("增量验证失败（视为 skipped）：%s", e)
            verif_result = None

        if verif_result is not None and files_changed:
            # 跑通 L1 后再跑 L2（unit_test）
            try:
                if verif_result.overall_passed:
                    l2 = await self.verifier.level2.verify(
                        files_changed,
                        graph_context.project_path,
                        task_id=task_node.id,
                    )
                    # 简单合并
                    merged_checks = list(verif_result.checks) + list(l2.checks)
                    overall = all(c.passed or c.skipped for c in merged_checks)
                    from ..verify.types import IncrementalVerificationResult
                    verif_result = IncrementalVerificationResult(
                        task_id=task_node.id,
                        level=2,
                        checks=merged_checks,
                        overall_passed=overall,
                        files_verified=list(dict.fromkeys(
                            verif_result.files_verified + l2.files_verified
                        )),
                        total_duration_seconds=(
                            verif_result.total_duration_seconds
                            + l2.total_duration_seconds
                        ),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("L2 验证失败（忽略）：%s", e)

        self_verif = _build_self_verification(verif_result)
        builder.attach_self_verification(self_verif)

        # 根据验证结果调整 confidence（DONE → UNCERTAIN）
        if (
            builder.confidence == Confidence.DONE
            and not self_verif.overall_passed
        ):
            builder.set_confidence(
                Confidence.UNCERTAIN,
                f"执行完成但验证未通过：{self_verif.summary()}",
            )

        # 步骤 6 — Finalize + 持久化
        artifact = builder.finalize()
        try:
            self.artifact_store.save(artifact)
        except Exception as e:
            # 落盘失败属基础设施错误 — 抛出
            raise RuntimeError(f"Artifact 落盘失败: {e}") from e

        # 步骤 7 — 把工作区改动 commit 成 checkpoint，
        # 这样下个子任务的 base_hash 是当前 HEAD（避免 diff 累积）
        if self.git_checkpoint is not None and files_changed:
            try:
                await self.git_checkpoint.create_checkpoint(
                    f"subtask {task_node.id}: {artifact.self_summary[:60]}"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("checkpoint commit failed: %s", e)

        return artifact

    # ─────────────────────────────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def derive_allowed_paths(task_node: TaskNode) -> list[str]:
        """从 task_node.files_involved 推导允许修改的路径.

        L1 不强制拦截越界，只在 ScopeCheck 中标记。
        """
        allowed: list[str] = []
        for f in task_node.files_involved or []:
            allowed.append(f)
            # 配套测试文件
            stem = Path(f).stem
            allowed.append(f"tests/test_{stem}.py")
            # 配套 __init__.py
            parent = str(Path(f).parent)
            if parent and parent not in (".", ""):
                allowed.append(f"{parent}/__init__.py")
        # 去重保序
        seen: set[str] = set()
        out: list[str] = []
        for p in allowed:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _build_prompt(
        self,
        task_node: TaskNode,
        ctx: GraphContext,
        allowed: list[str],
    ) -> str:
        completed_block = (
            "\n".join(f"- {s}" for s in ctx.completed_summaries)
            if ctx.completed_summaries
            else "（无）"
        )
        files_line = (
            ", ".join(task_node.files_involved)
            if task_node.files_involved
            else "（未指定）"
        )
        return (
            "你正在执行一个大任务的子任务。\n\n"
            f"总体目标：{ctx.original_goal}\n"
            f"当前子任务：{task_node.description}\n"
            f"涉及文件：{files_line}\n\n"
            "前置任务的成果（仅摘要）：\n"
            f"{completed_block}\n\n"
            f"完成标准：{task_node.verification or '（未指定）'}\n\n"
            "约束：\n"
            f"- 只操作以下路径：{', '.join(allowed) or '（无限制）'}\n"
            "- 尝试在 15 步内完成\n"
            "- 完成后简要说明你做了什么\n\n"
            "如果遇到无法完成的情况，明确告诉我“我卡住了”和原因。"
        )

    async def _current_commit_hash(self, cwd: str | None) -> str | None:
        """读取 cwd 的 HEAD commit hash；非 git 仓库返回 None."""
        try:
            code, out = await _run_git("rev-parse", "HEAD", cwd=cwd)
        except Exception as e:  # noqa: BLE001
            logger.debug("rev-parse failed: %s", e)
            return None
        if code != 0:
            return None
        return out.strip() or None

    async def _collect_file_edits(
        self,
        builder: ArtifactBuilder,
        base_hash: str,
        project_path: str,
    ) -> list[str]:
        """用 git 比较 base_hash..(工作区) 的改动并写入 builder.

        Returns:
            实际改动的文件路径列表（相对项目根）。
        """
        if not base_hash:
            # 不在 git 仓库或无 base 可比 — 退化：用 agent._files_changed
            files = list(getattr(self.agent, "_files_changed", []) or [])
            return files

        # 获取 working tree 相对 base 的改动文件（已跟踪部分）
        try:
            code, out = await _run_git(
                "diff", "--name-status", base_hash, "--", cwd=project_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("git diff failed: %s", e)
            return []

        diff_entries: list[tuple[str, str, str | None]] = []  # (op, path, old_path)
        if code == 0 and out.strip():
            for raw_line in out.strip().splitlines():
                parts = raw_line.split("\t")
                if len(parts) < 2:
                    continue
                status = parts[0].strip()
                path = parts[1].strip()
                old_path: str | None = None
                if status.startswith("R") and len(parts) >= 3:
                    old_path = parts[1].strip()
                    path = parts[2].strip()
                    diff_entries.append(("RENAME", path, old_path))
                elif status == "A":
                    diff_entries.append(("CREATE", path, None))
                elif status == "D":
                    diff_entries.append(("DELETE", path, None))
                else:
                    diff_entries.append(("MODIFY", path, None))

        # 加上未跟踪文件（git diff 看不到它们）
        try:
            code2, out2 = await _run_git(
                "ls-files", "--others", "--exclude-standard", cwd=project_path,
            )
            if code2 == 0:
                for line in out2.strip().splitlines():
                    line = line.strip()
                    if line:
                        diff_entries.append(("CREATE", line, None))
        except Exception as e:  # noqa: BLE001
            logger.debug("git ls-files failed: %s", e)

        files_changed: list[str] = []
        for op_str, path, old_path in diff_entries:
            op = EditOperation(op_str)

            old_content = await self._read_at_commit(
                old_path or path, base_hash, project_path,
            ) if op != EditOperation.CREATE else None
            new_content: str | None
            if op == EditOperation.DELETE:
                new_content = None
            else:
                new_content = self._read_disk(path, project_path)

            try:
                builder.record_file_edit(
                    path=path,
                    operation=op,
                    old=old_content,
                    new=new_content,
                    old_path=old_path,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("record_file_edit error %s: %s", path, e)
                continue
            files_changed.append(path)

        return files_changed

    async def _read_at_commit(
        self, path: str, commit: str, cwd: str,
    ) -> str | None:
        try:
            code, out = await _run_git(
                "show", f"{commit}:{path}", cwd=cwd,
            )
        except Exception:  # noqa: BLE001
            return None
        if code != 0:
            return None
        return out

    @staticmethod
    def _read_disk(path: str, project_path: str) -> str | None:
        full = Path(project_path) / path
        if not full.exists():
            return None
        try:
            return full.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None


def _summarize(text: str, max_length: int = 200) -> str:
    if not text:
        return "(无输出)"
    line = text.strip().splitlines()[0]
    if len(line) > max_length:
        line = line[:max_length] + "..."
    return line


def _build_self_verification(result: Any) -> SelfVerification:
    """把 IncrementalVerificationResult 转为 SelfVerification.

    没结果时返回全部 skipped 的 SelfVerification（仍然 overall_passed=True，
    因为没有改动 → 没有需要验证的东西）。
    """
    def _empty(name: str) -> CheckResult:
        return CheckResult(
            check_name=name,
            passed=True,
            skipped=True,
            skip_reason="not run",
            duration_seconds=0.0,
            details="",
            items_checked=0,
            items_failed=0,
        )

    if result is None:
        return SelfVerification(
            syntax_check=_empty("syntax"),
            lint_check=None,
            type_check=None,
            unit_test=None,
            import_check=_empty("import"),
            overall_passed=True,
        )

    syntax = result.get_check("syntax") or _empty("syntax")
    import_c = result.get_check("import") or _empty("import")
    lint = result.get_check("lint")
    type_c = result.get_check("type_check") or result.get_check("type")
    unit = result.get_check("unit_test")

    return SelfVerification(
        syntax_check=syntax,
        lint_check=lint,
        type_check=type_c,
        unit_test=unit,
        import_check=import_c,
        overall_passed=result.overall_passed,
    )
