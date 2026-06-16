"""Goal-Driven 模式集成测试 — 状态机循环、Ledger 持久化、中断恢复.

运行:
    uv run pytest tests/test_goal_integration.py -xvs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from mini_code_agent.cli.repl import REPL
from mini_code_agent.core.agent import Agent
from mini_code_agent.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    StreamDelta,
    StreamDeltaType,
    TokenUsage,
    ToolCall,
    ToolParam,
)
from mini_code_agent.longrun import TaskLedgerManager
from mini_code_agent.longrun.ledger_types import TaskRunStatus
from mini_code_agent.safety import CommandFilter, FileGuard, LoopGuard
from mini_code_agent.tools.base import ToolRegistry
from mini_code_agent.tools.file_ops import ReadFileTool


# ---------------------------------------------------------------------------
# Mock Streaming LLM Client
# ---------------------------------------------------------------------------


class MockStreamingLLMClient(LLMClient):
    """可按轮次返回预设 StreamDelta 序列的 Mock."""

    def __init__(self, turn_responses: list[list[StreamDelta]]) -> None:
        super().__init__(model="mock-stream")
        self._turns = list(turn_responses)
        self._call_index = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        raise NotImplementedError

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamDelta]:
        if self._call_index >= len(self._turns):
            # 安全兜底：返回 complete 防止无限循环
            yield StreamDelta(type=StreamDeltaType.TEXT, content="[goal_status: complete]")
            yield StreamDelta(type=StreamDeltaType.FINISH, usage=TokenUsage(10, 5))
            return
        deltas = self._turns[self._call_index]
        self._call_index += 1
        for d in deltas:
            yield d


class InterruptOnTurnClient(LLMClient):
    """在指定轮次的 chat_stream 中抛 KeyboardInterrupt."""

    def __init__(self, normal_turns: list[list[StreamDelta]], interrupt_on: int) -> None:
        super().__init__(model="mock-interrupt")
        self._turns = list(normal_turns)
        self._interrupt_on = interrupt_on
        self._call_index = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        raise NotImplementedError

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamDelta]:
        idx = self._call_index
        self._call_index += 1
        if idx == self._interrupt_on:
            raise KeyboardInterrupt
        if idx < len(self._turns):
            for d in self._turns[idx]:
                yield d
        else:
            yield StreamDelta(type=StreamDeltaType.TEXT, content="[goal_status: complete]")
            yield StreamDelta(type=StreamDeltaType.FINISH, usage=TokenUsage(10, 5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_turn(text: str, tokens_in: int = 100, tokens_out: int = 50) -> list[StreamDelta]:
    return [
        StreamDelta(type=StreamDeltaType.TEXT, content=text),
        StreamDelta(type=StreamDeltaType.FINISH, usage=TokenUsage(tokens_in, tokens_out)),
    ]


def _build_repl(llm_client: LLMClient, tmp_path: Path) -> tuple[REPL, TaskLedgerManager]:
    """构建带有 ledger 的最小 REPL 实例."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    os.chdir(str(project_dir))

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    ledger_dir = str(project_dir / ".agent" / "ledger")
    ledger_manager = TaskLedgerManager(storage_dir=ledger_dir)

    agent = Agent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt="test",
        command_filter=CommandFilter(),
        file_guard=FileGuard(work_dir=project_dir),
        loop_guard=LoopGuard(),
        ledger_manager=ledger_manager,
    )

    from rich.console import Console
    repl = REPL(agent=agent, console=Console(quiet=True))
    return repl, ledger_manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_completes_after_two_turns(tmp_path: Path) -> None:
    """master 两轮后输出 [goal_status: complete]，验证 ledger 状态."""
    client = MockStreamingLLMClient([
        _make_text_turn("正在分析项目...\n[goal_status: active]"),
        _make_text_turn("所有标准已达成。\n[goal_status: complete]"),
    ])
    repl, manager = _build_repl(client, tmp_path)

    # mock prompt_async 直接返回 criteria
    with patch.object(repl._prompt_session, "prompt_async", return_value="测试通过"):
        await repl._handle_goal_command("测试目标")

    ledgers = manager.list_all()
    assert len(ledgers) >= 1
    ledger = manager.load(ledgers[0].task_id)
    assert ledger.status == TaskRunStatus.COMPLETED
    assert ledger.current_phase == "done"
    assert ledger.total_tokens_used > 0
    nodes = ledger.task_graph_snapshot.get("nodes", {})
    assert nodes.get("goal", {}).get("status") == "completed"


@pytest.mark.asyncio
async def test_goal_blocked_stops_loop(tmp_path: Path) -> None:
    """首轮返回 blocked，循环应立即停止."""
    client = MockStreamingLLMClient([
        _make_text_turn("需要用户决策，无法继续。\n[goal_status: blocked]"),
    ])
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="OK"):
        await repl._handle_goal_command("被阻塞的目标")

    ledgers = manager.list_all()
    ledger = manager.load(ledgers[0].task_id)
    assert ledger.status == TaskRunStatus.PAUSED
    assert ledger.current_phase == "blocked"
    # mock 只跑了 1 轮
    assert client._call_index == 1


@pytest.mark.asyncio
async def test_goal_interrupt_saves_paused(tmp_path: Path) -> None:
    """第二轮抛 KeyboardInterrupt，验证 ledger 保存为 PAUSED."""
    client = InterruptOnTurnClient(
        normal_turns=[_make_text_turn("第一步完成\n[goal_status: active]")],
        interrupt_on=1,
    )
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="criteria"):
        await repl._handle_goal_command("中断测试")

    ledgers = manager.list_all()
    ledger = manager.load(ledgers[0].task_id)
    assert ledger.status == TaskRunStatus.PAUSED
    assert ledger.current_phase == "paused"
    assert ledger.total_tokens_used > 0
    assert ledger.total_wall_time_seconds > 0


@pytest.mark.asyncio
async def test_goal_status_command(tmp_path: Path) -> None:
    """调用 _show_goal_status 不报错，且能显示 ledger 信息."""
    client = MockStreamingLLMClient([
        _make_text_turn("Done\n[goal_status: complete]"),
    ])
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="pass"):
        await repl._handle_goal_command("status 测试")

    # 现在应该有 ledger 了
    repl._show_goal_status()  # 不抛异常即可


@pytest.mark.asyncio
async def test_goal_cancel_command(tmp_path: Path) -> None:
    """创建 paused ledger 后 cancel，验证 status == FAILED."""
    client = InterruptOnTurnClient(
        normal_turns=[_make_text_turn("开始\n[goal_status: active]")],
        interrupt_on=1,
    )
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="criteria"):
        await repl._handle_goal_command("取消测试")

    # 此时应该是 paused
    ledgers = manager.list_all()
    ledger = manager.load(ledgers[0].task_id)
    assert ledger.status == TaskRunStatus.PAUSED

    # cancel
    repl._goal_cancel()

    ledger = manager.load(ledgers[0].task_id)
    assert ledger.status == TaskRunStatus.FAILED
    assert ledger.current_phase == "cancelled"


@pytest.mark.asyncio
async def test_goal_resume_retriggers(tmp_path: Path) -> None:
    """resume 暂停的 goal 后重新执行并 complete."""
    # 第一次执行：中断
    client = InterruptOnTurnClient(
        normal_turns=[_make_text_turn("开始\n[goal_status: active]")],
        interrupt_on=1,
    )
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="criteria"):
        await repl._handle_goal_command("恢复测试")

    ledgers = manager.list_all()
    assert ledgers[0].status == TaskRunStatus.PAUSED

    # 替换 client 为能 complete 的
    repl.agent.llm_client = MockStreamingLLMClient([
        _make_text_turn("恢复后完成\n[goal_status: complete]"),
    ])

    with patch.object(repl._prompt_session, "prompt_async", return_value="criteria"):
        await repl._goal_resume()

    # 应该有新的 completed ledger
    all_ledgers = manager.list_all()
    statuses = [l.status for l in all_ledgers]
    assert TaskRunStatus.COMPLETED in statuses


@pytest.mark.asyncio
async def test_ledger_json_persisted(tmp_path: Path) -> None:
    """验证 ledger JSON 文件实际写入磁盘且可解析."""
    client = MockStreamingLLMClient([
        _make_text_turn("Done\n[goal_status: complete]"),
    ])
    repl, manager = _build_repl(client, tmp_path)

    with patch.object(repl._prompt_session, "prompt_async", return_value="pass"):
        await repl._handle_goal_command("持久化测试")

    # 检查磁盘文件
    ledger_dir = tmp_path / "project" / ".agent" / "ledger"
    json_files = list(ledger_dir.glob("*.json"))
    assert len(json_files) >= 1

    with open(json_files[0]) as f:
        data = json.load(f)
    assert data["status"] == "COMPLETED"
    assert data["current_phase"] == "done"
    assert data["total_tokens_used"] > 0
