"""记忆系统测试.

包含：
  - ConversationManager: token 计数、压缩逻辑
  - ProjectMemory: 读写持久化、增删查
  - 记忆工具: AddMemoryTool、RecallMemoryTool

运行:
  uv run pytest tests/test_memory.py -xvs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_code_agent.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolParam,
)
from mini_code_agent.memory.conversation import ConversationManager, KEEP_RECENT_ROUNDS
from mini_code_agent.memory.project_memory import ProjectMemory


# ==================================================================
# 辅助：Mock LLM 客户端（用于压缩测试）
# ==================================================================


class MockSummaryClient(LLMClient):
    """始终返回固定摘要的 Mock 客户端."""

    def __init__(self, summary: str = "这是对话摘要") -> None:
        super().__init__(model="mock")
        self._summary = summary

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=self._summary, usage=TokenUsage(10, 10))

    def chat_stream(self, messages, tools=None):
        raise NotImplementedError


# ==================================================================
# ConversationManager 测试
# ==================================================================


class TestConversationManager:
    """ConversationManager 单元测试."""

    def test_init_system(self):
        """初始化 system prompt."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client)
        cm.init_system("你是助手")

        assert len(cm.messages) == 1
        assert cm.messages[0].content == "你是助手"
        assert cm.token_count > 0

    def test_append_updates_token_count(self):
        """append 消息后 token 计数增加."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client)
        cm.init_system("你是助手")

        before = cm.token_count
        cm.append(Message.user("你好"))
        assert cm.token_count > before
        assert len(cm.messages) == 2

    def test_needs_compression_below_threshold(self):
        """token 数未超阈值时不需要压缩."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client, max_tokens=100_000)
        cm.init_system("你是助手")
        cm.append(Message.user("短消息"))

        assert not cm.needs_compression()

    def test_needs_compression_above_threshold(self):
        """token 数超阈值时需要压缩."""
        client = MockSummaryClient()
        # 设置很低的 max_tokens，让少量消息就能触发
        cm = ConversationManager(llm_client=client, max_tokens=50)
        cm.init_system("你是助手")

        # 添加足够多的消息让 token 数超过 50 * 0.7 = 35
        for i in range(20):
            cm.append(Message.user(f"这是第 {i} 条很长的测试消息，包含一些额外内容"))
            cm.append(Message.assistant(f"这是第 {i} 条回复"))

        assert cm.needs_compression()

    @pytest.mark.asyncio
    async def test_compress_generates_summary(self):
        """压缩执行后，早期消息被摘要替代."""
        summary_text = "用户讨论了文件操作和代码修改"
        client = MockSummaryClient(summary=summary_text)

        # 设置很低的 max_tokens 以触发压缩
        cm = ConversationManager(llm_client=client, max_tokens=50)
        cm.init_system("你是助手")

        # 添加超过 KEEP_RECENT_ROUNDS 轮的对话
        for i in range(KEEP_RECENT_ROUNDS + 5):
            cm.append(Message.user(f"用户消息 {i}"))
            cm.append(Message.assistant(f"助手回复 {i}"))

        original_count = len(cm.messages)
        assert cm.needs_compression()

        compressed = await cm.compress()
        assert compressed is True

        # 压缩后消息应该减少了
        assert len(cm.messages) < original_count

        # 第一条仍然是 system prompt
        assert cm.messages[0].content == "你是助手"

        # 第二条应该是摘要
        assert "[对话历史摘要]" in cm.messages[1].content
        assert summary_text in cm.messages[1].content

    @pytest.mark.asyncio
    async def test_compress_preserves_recent_messages(self):
        """压缩后保留最近的对话轮."""
        client = MockSummaryClient(summary="摘要")
        cm = ConversationManager(llm_client=client, max_tokens=50)
        cm.init_system("你是助手")

        # 添加 KEEP_RECENT_ROUNDS + 3 轮
        total_rounds = KEEP_RECENT_ROUNDS + 3
        for i in range(total_rounds):
            cm.append(Message.user(f"用户消息 {i}"))
            cm.append(Message.assistant(f"助手回复 {i}"))

        await cm.compress()

        # 最后一条应该是最新的 assistant 消息
        assert cm.messages[-1].content == f"助手回复 {total_rounds - 1}"

    @pytest.mark.asyncio
    async def test_compress_skipped_when_not_needed(self):
        """不需要压缩时 compress 返回 False."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client, max_tokens=100_000)
        cm.init_system("你是助手")
        cm.append(Message.user("短消息"))

        compressed = await cm.compress()
        assert compressed is False

    @pytest.mark.asyncio
    async def test_compress_skipped_when_too_few_messages(self):
        """消息太少无法分离出早期消息时不压缩."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client, max_tokens=10)
        cm.init_system("你是助手")
        # 只加 2 轮，少于 KEEP_RECENT_ROUNDS
        cm.append(Message.user("消息1"))
        cm.append(Message.assistant("回复1"))

        compressed = await cm.compress()
        assert compressed is False

    def test_reset(self):
        """reset 清空历史并重新初始化."""
        client = MockSummaryClient()
        cm = ConversationManager(llm_client=client)
        cm.init_system("旧 prompt")
        cm.append(Message.user("消息"))
        cm.append(Message.assistant("回复"))

        cm.reset("新 prompt")

        assert len(cm.messages) == 1
        assert cm.messages[0].content == "新 prompt"

    def test_find_split_index(self):
        """_find_split_index 正确计算分界点."""
        messages = [
            Message.user("u1"),
            Message.assistant("a1"),
            Message.user("u2"),
            Message.assistant("a2"),
            Message.user("u3"),
            Message.assistant("a3"),
        ]
        # 保留最近 2 轮 → 分界在第 2 个 user（index 2）
        idx = ConversationManager._find_split_index(messages, keep_rounds=2)
        assert idx == 2
        assert messages[idx].content == "u2"

    def test_find_split_index_with_tool_messages(self):
        """tool 消息不算一轮."""
        from mini_code_agent.llm.base import ToolResult as LLMToolResult

        messages = [
            Message.user("u1"),
            Message.assistant("a1", tool_calls=[ToolCall(id="c1", name="read", arguments={})]),
            Message.tool(LLMToolResult(tool_call_id="c1", content="file content")),
            Message.user("u2"),
            Message.assistant("a2"),
        ]
        # 保留最近 1 轮 → 分界在 index 3 (u2)
        idx = ConversationManager._find_split_index(messages, keep_rounds=1)
        assert idx == 3


# ==================================================================
# ProjectMemory 测试
# ==================================================================


class TestProjectMemory:
    """ProjectMemory 单元测试."""

    def test_empty_project_memory(self, tmp_path: Path):
        """新项目没有记忆数据."""
        pm = ProjectMemory(tmp_path)
        assert pm.data.conventions == []
        assert pm.data.decisions == []
        assert pm.data.known_issues == []

    def test_add_convention(self, tmp_path: Path):
        """添加约定并持久化."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest 做测试")

        assert "使用 pytest 做测试" in pm.data.conventions

        # 验证持久化
        pm2 = ProjectMemory(tmp_path)
        assert "使用 pytest 做测试" in pm2.data.conventions

    def test_add_convention_dedup(self, tmp_path: Path):
        """重复约定不会重复添加."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest")
        pm.add_convention("使用 pytest")

        assert pm.data.conventions.count("使用 pytest") == 1

    def test_add_decision(self, tmp_path: Path):
        """添加技术决策."""
        pm = ProjectMemory(tmp_path)
        pm.add_decision("选择 SQLite", "轻量级，不需要单独部署数据库")

        assert len(pm.data.decisions) == 1
        assert pm.data.decisions[0].decision == "选择 SQLite"
        assert "轻量级" in pm.data.decisions[0].reason

    def test_add_known_issue(self, tmp_path: Path):
        """添加已知问题."""
        pm = ProjectMemory(tmp_path)
        pm.add_known_issue("import 循环依赖", "使用延迟 import")

        assert len(pm.data.known_issues) == 1
        assert pm.data.known_issues[0].issue == "import 循环依赖"

    def test_recall_by_keyword(self, tmp_path: Path):
        """按关键词搜索记忆."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest 做测试")
        pm.add_convention("commit message 用英文")
        pm.add_decision("选择 SQLite", "轻量级")
        pm.add_known_issue("pytest 收集慢", "用 --co 提前检查")

        results = pm.recall("pytest")
        assert len(results) == 2  # 约定 + 已知问题

        results = pm.recall("SQLite")
        assert len(results) == 1
        assert "[决策" in results[0]

    def test_recall_case_insensitive(self, tmp_path: Path):
        """关键词搜索不区分大小写."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 TypeScript")

        results = pm.recall("typescript")
        assert len(results) == 1

    def test_recall_no_match(self, tmp_path: Path):
        """搜索不到时返回空列表."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest")

        results = pm.recall("不存在的关键词")
        assert results == []

    def test_format_for_prompt_empty(self, tmp_path: Path):
        """空记忆返回空字符串."""
        pm = ProjectMemory(tmp_path)
        assert pm.format_for_prompt() == ""

    def test_format_for_prompt_with_data(self, tmp_path: Path):
        """有数据时格式化为 prompt 文本."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest")
        pm.add_decision("选择 SQLite", "轻量级")

        text = pm.format_for_prompt()
        assert "项目约定" in text
        assert "使用 pytest" in text
        assert "技术决策" in text
        assert "SQLite" in text

    def test_persistence_roundtrip(self, tmp_path: Path):
        """完整的持久化往返测试."""
        pm = ProjectMemory(tmp_path)
        pm.add_convention("约定1")
        pm.add_decision("决策1", "原因1")
        pm.add_known_issue("问题1", "解法1")

        # 从磁盘重新加载
        pm2 = ProjectMemory(tmp_path)
        assert pm2.data.conventions == ["约定1"]
        assert pm2.data.decisions[0].decision == "决策1"
        assert pm2.data.known_issues[0].solution == "解法1"

    def test_corrupted_json_fallback(self, tmp_path: Path):
        """JSON 损坏时回退到空记忆."""
        memory_dir = tmp_path / ".agent"
        memory_dir.mkdir()
        (memory_dir / "memory.json").write_text("not valid json", encoding="utf-8")

        pm = ProjectMemory(tmp_path)
        assert pm.data.conventions == []


# ==================================================================
# 记忆工具测试
# ==================================================================


class TestMemoryTools:
    """AddMemoryTool / RecallMemoryTool 测试."""

    @pytest.mark.asyncio
    async def test_add_memory_convention(self, tmp_path: Path):
        from mini_code_agent.tools.memory import AddMemoryTool

        pm = ProjectMemory(tmp_path)
        tool = AddMemoryTool()
        tool._project_memory = pm

        result = await tool.execute(type="convention", content="使用 black 格式化")
        assert not result.is_error
        assert "使用 black 格式化" in pm.data.conventions

    @pytest.mark.asyncio
    async def test_add_memory_decision(self, tmp_path: Path):
        from mini_code_agent.tools.memory import AddMemoryTool

        pm = ProjectMemory(tmp_path)
        tool = AddMemoryTool()
        tool._project_memory = pm

        result = await tool.execute(
            type="decision", content="用 Redis 做缓存", reason="性能要求"
        )
        assert not result.is_error
        assert pm.data.decisions[0].decision == "用 Redis 做缓存"

    @pytest.mark.asyncio
    async def test_add_memory_known_issue(self, tmp_path: Path):
        from mini_code_agent.tools.memory import AddMemoryTool

        pm = ProjectMemory(tmp_path)
        tool = AddMemoryTool()
        tool._project_memory = pm

        result = await tool.execute(
            type="known_issue", content="OOM on large files", solution="streaming read"
        )
        assert not result.is_error
        assert pm.data.known_issues[0].issue == "OOM on large files"

    @pytest.mark.asyncio
    async def test_add_memory_no_project_memory(self):
        from mini_code_agent.tools.memory import AddMemoryTool

        tool = AddMemoryTool()
        result = await tool.execute(type="convention", content="test")
        assert result.is_error

    @pytest.mark.asyncio
    async def test_recall_memory(self, tmp_path: Path):
        from mini_code_agent.tools.memory import RecallMemoryTool

        pm = ProjectMemory(tmp_path)
        pm.add_convention("使用 pytest 做测试")
        pm.add_convention("使用 mypy 做类型检查")

        tool = RecallMemoryTool()
        tool._project_memory = pm

        result = await tool.execute(keyword="pytest")
        assert not result.is_error
        assert "pytest" in result.output

    @pytest.mark.asyncio
    async def test_recall_memory_no_match(self, tmp_path: Path):
        from mini_code_agent.tools.memory import RecallMemoryTool

        pm = ProjectMemory(tmp_path)
        tool = RecallMemoryTool()
        tool._project_memory = pm

        result = await tool.execute(keyword="不存在")
        assert not result.is_error
        assert "未找到" in result.output
