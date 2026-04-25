"""对话历史管理 — token 计数、自动压缩."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..context.context_builder import estimate_tokens
from ..llm.base import LLMClient, Message, Role, TokenUsage

logger = logging.getLogger(__name__)

# 压缩时发送给 LLM 的 prompt
SUMMARIZE_PROMPT = (
    "请总结以下对话历史的关键信息，包括：\n"
    "用户的目标、已完成的修改、发现的问题、做出的决策。\n"
    "保留文件路径、函数名等具体信息。简洁但不要丢失关键细节。"
)

# 保留最近的对话轮数（user+assistant 算一轮）
KEEP_RECENT_ROUNDS = 10


@dataclass
class ConversationManager:
    """管理对话历史，支持 token 计数和自动压缩.

    压缩策略（当 token 数超过阈值的 70% 时触发）：
      1. 保留 system prompt（不动）
      2. 保留最近 KEEP_RECENT_ROUNDS 轮对话（不动）
      3. 把更早的对话调用 LLM 生成摘要
      4. 用摘要替换原始消息
    """

    llm_client: LLMClient
    max_tokens: int = 100_000  # token 上限
    compress_ratio: float = 0.7  # 超过 max_tokens * ratio 时触发压缩

    _messages: list[Message] = field(default_factory=list, repr=False)
    _token_count: int = field(default=0, repr=False)

    @property
    def messages(self) -> list[Message]:
        """返回当前消息列表（只读视图）."""
        return self._messages

    @property
    def token_count(self) -> int:
        """当前消息的估算 token 总数."""
        return self._token_count

    def init_system(self, system_prompt: str) -> None:
        """初始化 system prompt，清空之前的历史."""
        self._messages = [Message.system(system_prompt)]
        self._token_count = estimate_tokens(system_prompt)

    def update_system(self, system_prompt: str) -> None:
        """就地替换 system prompt（不影响对话历史）."""
        if self._messages and self._messages[0].role == Role.SYSTEM:
            old_tokens = self._estimate_message_tokens(self._messages[0])
            self._messages[0] = Message.system(system_prompt)
            new_tokens = self._estimate_message_tokens(self._messages[0])
            self._token_count += new_tokens - old_tokens
        else:
            # 没有 system prompt，直接插入
            msg = Message.system(system_prompt)
            self._messages.insert(0, msg)
            self._token_count += self._estimate_message_tokens(msg)

    def append(self, message: Message) -> None:
        """追加一条消息并更新 token 计数."""
        self._messages.append(message)
        self._token_count += self._estimate_message_tokens(message)

    def reset(self, system_prompt: str) -> None:
        """重置对话历史（保留新的 system prompt）."""
        self.init_system(system_prompt)

    def needs_compression(self) -> bool:
        """判断是否需要压缩."""
        return self._token_count > self.max_tokens * self.compress_ratio

    async def compress(self) -> bool:
        """执行压缩：把早期对话摘要化.

        Returns:
            True 如果执行了压缩，False 如果不需要压缩或消息太少。
        """
        if not self.needs_compression():
            return False

        # 找到 system prompt（第一条）
        if not self._messages or self._messages[0].role != Role.SYSTEM:
            return False

        system_msg = self._messages[0]
        rest = self._messages[1:]

        # 找到最近 KEEP_RECENT_ROUNDS 轮对话的分界点
        split_idx = self._find_split_index(rest, KEEP_RECENT_ROUNDS)

        if split_idx <= 0:
            # 没有可压缩的早期消息
            return False

        old_messages = rest[:split_idx]
        recent_messages = rest[split_idx:]

        # 调用 LLM 生成摘要
        summary = await self._generate_summary(old_messages)
        if not summary:
            return False

        # 用摘要替换原始消息
        summary_msg = Message.user(f"[对话历史摘要]\n{summary}")

        old_token_count = self._token_count
        self._messages = [system_msg, summary_msg] + recent_messages
        self._token_count = self._recount_tokens()

        logger.info(
            "对话压缩完成: %d tokens → %d tokens (压缩了 %d 条消息)",
            old_token_count,
            self._token_count,
            len(old_messages),
        )
        return True

    async def _generate_summary(self, messages: list[Message]) -> str:
        """调用 LLM 把一段对话历史压缩成摘要."""
        # 把需要摘要的消息格式化成文本
        conversation_text = self._format_messages_for_summary(messages)

        summary_messages = [
            Message.system(SUMMARIZE_PROMPT),
            Message.user(conversation_text),
        ]

        try:
            response = await self.llm_client.chat(
                messages=summary_messages,
                tools=None,
            )
            return response.content.strip()
        except Exception as e:
            logger.error("生成对话摘要失败: %s", e)
            return ""

    @staticmethod
    def _find_split_index(messages: list[Message], keep_rounds: int) -> int:
        """从后往前数 keep_rounds 轮 user 消息，返回分界索引.

        一轮 = 一条 user 消息 + 后续的 assistant/tool 消息。
        """
        user_count = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == Role.USER:
                user_count += 1
                if user_count >= keep_rounds:
                    return i
        # 消息轮数不足 keep_rounds，无法压缩
        return 0

    @staticmethod
    def _format_messages_for_summary(messages: list[Message]) -> str:
        """把消息列表格式化成易于 LLM 理解的文本."""
        lines: list[str] = []
        for msg in messages:
            if msg.role == Role.USER:
                lines.append(f"用户: {msg.content or ''}")
            elif msg.role == Role.ASSISTANT:
                content = msg.content or ""
                if msg.tool_calls:
                    tool_names = ", ".join(tc.name for tc in msg.tool_calls)
                    content += f" [调用了工具: {tool_names}]"
                lines.append(f"助手: {content}")
            elif msg.role == Role.TOOL:
                if msg.tool_result:
                    # 截断过长的工具输出
                    output = msg.tool_result.content
                    if len(output) > 500:
                        output = output[:497] + "..."
                    lines.append(f"工具结果: {output}")
        return "\n".join(lines)

    @staticmethod
    def _estimate_message_tokens(message: Message) -> int:
        """估算单条消息的 token 数."""
        tokens = 0
        if message.content:
            tokens += estimate_tokens(message.content)
        for tc in message.tool_calls:
            tokens += estimate_tokens(tc.name)
            tokens += estimate_tokens(tc.arguments_json())
        if message.tool_result:
            tokens += estimate_tokens(message.tool_result.content)
        # 角色标签等开销
        tokens += 4
        return tokens

    def _recount_tokens(self) -> int:
        """重新计算所有消息的 token 总数."""
        return sum(self._estimate_message_tokens(m) for m in self._messages)
