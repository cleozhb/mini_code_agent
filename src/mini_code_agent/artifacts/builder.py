"""ArtifactBuilder — 增量构建 Artifact 的辅助工具.

让 Worker / Agent 在执行过程中"一点点往 Artifact 里塞东西"，
而不是在最后一次性构造。
"""

from __future__ import annotations

import difflib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .artifact import ArtifactStatus, Confidence, SubtaskArtifact
from .decision import ArtifactDecision
from .patch import EditOperation, FileEdit, Patch, generate_unified_diff
from .resource import ResourceUsage
from .scope import ScopeCheck, ScopeChecker
from .verification import SelfVerification


class ArtifactBuilderError(Exception):
    """ArtifactBuilder 构建过程中的错误."""


class ArtifactBuilder:
    """增量构建 SubtaskArtifact 的辅助工具."""

    def __init__(
        self,
        task_id: str,
        task_description: str,
        allowed_paths: list[str],
        verification_spec: str,
        producer: str,
    ) -> None:
        self._task_id = task_id
        self._task_description = task_description
        self._allowed_paths = allowed_paths
        self._verification_spec = verification_spec
        self._producer = producer

        self._artifact_id = str(uuid.uuid4())
        self._base_git_hash: str = ""
        self._start_time: float = 0.0

        # 累积数据
        self._edits: list[FileEdit] = []
        self._decisions: list[ArtifactDecision] = []
        self._self_verification: SelfVerification | None = None
        self._confidence: Confidence | None = None
        self._self_summary: str = ""
        self._open_questions: list[str] = []

        # 资源统计
        self._tokens_input: int = 0
        self._tokens_output: int = 0
        self._llm_calls: int = 0
        self._tool_calls: int = 0
        self._model_used: str = ""

        self._finalized = False

    def start(self, base_git_hash: str) -> None:
        """记录开始时间和基线 commit."""
        self._base_git_hash = base_git_hash
        self._start_time = time.monotonic()

    def record_file_edit(
        self,
        path: str,
        operation: EditOperation,
        old: str | None,
        new: str | None,
        old_path: str | None = None,
    ) -> None:
        """累积一个文件编辑，自动生成 unified_diff 和行统计."""
        self._check_not_finalized()

        diff = generate_unified_diff(path, operation, old, new, old_path)

        old_lines = (old or "").splitlines()
        new_lines = (new or "").splitlines()
        # 用 difflib 计算精确的增删行数
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        added = 0
        removed = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                removed += i2 - i1
                added += j2 - j1
            elif tag == "delete":
                removed += i2 - i1
            elif tag == "insert":
                added += j2 - j1

        self._edits.append(
            FileEdit(
                path=path,
                operation=operation,
                old_content=old,
                new_content=new,
                old_path=old_path,
                unified_diff=diff,
                lines_added=added,
                lines_removed=removed,
            )
        )

    def record_decision(
        self,
        description: str,
        reason: str,
        alternatives: list[str] | None = None,
        reversible: bool = True,
    ) -> None:
        """记录一个关键决策."""
        self._check_not_finalized()
        step = len(self._decisions) + 1
        self._decisions.append(
            ArtifactDecision(
                description=description,
                reason=reason,
                alternatives_considered=alternatives or [],
                reversible=reversible,
                step_number=step,
            )
        )

    def record_tool_call(self, name: str, input: str, output: str) -> None:
        """记录工具调用（只累计计数，不存内容）."""
        self._check_not_finalized()
        self._tool_calls += 1

    def record_llm_call(self, tokens_in: int, tokens_out: int, model: str) -> None:
        """记录 LLM 调用."""
        self._check_not_finalized()
        self._tokens_input += tokens_in
        self._tokens_output += tokens_out
        self._llm_calls += 1
        self._model_used = model

    def attach_self_verification(self, result: SelfVerification) -> None:
        """附加自验证结果."""
        self._check_not_finalized()
        self._self_verification = result

    def set_confidence(self, confidence: Confidence, summary: str) -> None:
        """设置信心等级和自我总结."""
        self._check_not_finalized()
        self._confidence = confidence
        self._self_summary = summary

    @property
    def confidence(self) -> Confidence | None:
        """当前已设置的 confidence；未设置返回 None."""
        return self._confidence

    def add_open_question(self, question: str) -> None:
        """添加一个遗留疑问."""
        self._check_not_finalized()
        self._open_questions.append(question)

    def finalize(self) -> SubtaskArtifact:
        """生成最终的 SubtaskArtifact，同时计算 scope_check."""
        self._check_not_finalized()

        if self._self_verification is None:
            raise ArtifactBuilderError("finalize 前必须 attach_self_verification")
        if self._confidence is None:
            raise ArtifactBuilderError("finalize 前必须 set_confidence")

        wall_time = time.monotonic() - self._start_time if self._start_time else 0.0

        # 构建 Patch
        total_added = sum(e.lines_added for e in self._edits)
        total_removed = sum(e.lines_removed for e in self._edits)
        patch = Patch(
            edits=list(self._edits),
            total_files_changed=len(self._edits),
            total_lines_added=total_added,
            total_lines_removed=total_removed,
            base_git_hash=self._base_git_hash,
        )

        # 计算 scope_check
        touched = [e.path for e in self._edits]
        scope_check = ScopeChecker.check(self._allowed_paths, touched)

        # 构建 ResourceUsage
        resource_usage = ResourceUsage(
            tokens_input=self._tokens_input,
            tokens_output=self._tokens_output,
            tokens_total=self._tokens_input + self._tokens_output,
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            wall_time_seconds=wall_time,
            model_used=self._model_used,
        )

        artifact = SubtaskArtifact(
            artifact_id=self._artifact_id,
            task_id=self._task_id,
            created_at=datetime.now(timezone.utc),
            producer=self._producer,
            task_description=self._task_description,
            allowed_paths=self._allowed_paths,
            verification_spec=self._verification_spec,
            patch=patch,
            self_verification=self._self_verification,
            scope_check=scope_check,
            decisions=list(self._decisions),
            resource_usage=resource_usage,
            confidence=self._confidence,
            self_summary=self._self_summary,
            open_questions=list(self._open_questions),
            status=ArtifactStatus.DRAFT,
        )

        self._finalized = True
        return artifact

    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise ArtifactBuilderError("Artifact 已经 finalize，不能再修改")
