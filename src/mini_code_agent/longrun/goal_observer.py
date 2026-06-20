"""Shared Goal mode ledger observer helpers."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from ..core.agent import Agent, AgentObserver
from ..longrun.ledger_manager import TaskLedgerManager
from ..longrun.ledger_types import CompletedTaskRecord, FailedAttemptRecord
from ..longrun.task_ledger import TaskLedger
from ..tools.base import ToolResult

_STOP_REASON_RE = re.compile(r"stop_reason:\s*([^\]\s]+)")
_TOKEN_RE = re.compile(r"\b(?:total|total_tokens)=(\d+)\b")


def parse_subagent_stop_reason(output: str) -> str:
    """Parse stop_reason from current SubAgent metadata output."""
    match = _STOP_REASON_RE.search(output)
    return match.group(1) if match else "unknown"


def parse_subagent_total_tokens(output: str) -> int:
    """Parse total token count from current or legacy SubAgent usage metadata."""
    for line in output.splitlines()[:3]:
        match = _TOKEN_RE.search(line)
        if match:
            return int(match.group(1))
    return 0


class GoalLedgerObserver(AgentObserver):
    """Record SubAgent outcomes and LLM usage into the goal ledger."""

    def __init__(
        self,
        ledger: TaskLedger,
        manager: TaskLedgerManager,
        files_changed_buffer: list[str],
    ) -> None:
        self._ledger = ledger
        self._manager = manager
        self._files_changed_buffer = files_changed_buffer
        self._sub_count = 0

    def on_tool_call(self, name: str, args: dict[str, Any], result: Any) -> None:
        if name != "SubAgent" or not isinstance(result, ToolResult):
            return

        self._sub_count += 1
        task_id = f"sub-{self._sub_count}"
        goal_desc = str(args.get("goal", ""))[:200]
        output = result.output or ""
        stop_reason = parse_subagent_stop_reason(output)
        token_count = parse_subagent_total_tokens(output)
        step_start = self._ledger.total_steps
        step_end = step_start + 1
        sub_failed = result.is_error or stop_reason not in {"ok", "unknown"}
        files_changed = list(self._files_changed_buffer)
        self._files_changed_buffer.clear()

        if sub_failed:
            reason = result.error or f"stop_reason={stop_reason}"
            self._ledger.failed_attempts.append(FailedAttemptRecord(
                task_id=task_id,
                artifact_id="",
                approach_description=goal_desc,
                failure_reason=reason,
                step_number=step_end,
            ))
        else:
            self._ledger.completed_tasks.append(CompletedTaskRecord(
                task_id=task_id,
                artifact_id="",
                description=goal_desc,
                self_summary=output[:300],
                files_changed=files_changed,
                verification_passed=False,
                confidence="DONE",
                step_number_start=step_start,
                step_number_end=step_end,
                token_count=token_count,
                timestamp=datetime.now(UTC),
            ))

        self._ledger.total_steps = step_end
        self._ledger.total_tokens_used += token_count
        self._ledger.token_budget_remaining = max(
            0, self._ledger.token_budget - self._ledger.total_tokens_used
        )
        self._manager.save(self._ledger)

    def on_llm_call(self, tokens_in: int, tokens_out: int, model: str) -> None:
        self._ledger.total_tokens_used += tokens_in + tokens_out
        self._ledger.token_budget_remaining = max(
            0, self._ledger.token_budget - self._ledger.total_tokens_used
        )
        self._manager.save(self._ledger)


def attach_goal_ledger_observer(
    agent: Agent,
    ledger: TaskLedger,
    manager: TaskLedgerManager,
    files_changed_buffer: list[str],
) -> None:
    agent.observers.append(GoalLedgerObserver(ledger, manager, files_changed_buffer))
