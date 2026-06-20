"""Checkpoint message serialization and restore helpers."""

from __future__ import annotations

from ..llm.base import Message, Role


def serialize_checkpoint_messages(messages: list[Message]) -> list[dict]:
    """Serialize non-system messages for checkpoint persistence."""
    return [m.to_dict() for m in messages if m.role != Role.SYSTEM]


def restore_checkpoint_messages(raw_messages: list[dict]) -> list[Message]:
    """Restore checkpoint messages and drop incomplete tool-call tails."""
    messages = [Message.from_dict(m) for m in raw_messages]
    messages = [m for m in messages if m.role != Role.SYSTEM]
    return repair_or_drop_unpaired_tool_messages(messages)


def repair_or_drop_unpaired_tool_messages(messages: list[Message]) -> list[Message]:
    """Drop orphan tool results and truncate an incomplete trailing tool call."""
    repaired: list[Message] = []
    pending: dict[str, int] = {}

    for message in messages:
        if message.role == Role.TOOL:
            tool_result = message.tool_result
            if tool_result is None or tool_result.tool_call_id not in pending:
                continue
            repaired.append(message)
            pending.pop(tool_result.tool_call_id, None)
            continue

        repaired.append(message)
        if message.role == Role.ASSISTANT and message.tool_calls:
            assistant_index = len(repaired) - 1
            for tool_call in message.tool_calls:
                pending[tool_call.id] = assistant_index

    if pending:
        truncate_from = min(pending.values())
        repaired = repaired[:truncate_from]

    return repaired
