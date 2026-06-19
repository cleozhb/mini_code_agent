"""运行时输入通道 — Agent core 只依赖 Protocol，不关心具体实现."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class InputKind(Enum):
    USER_INSTRUCTION = "user_instruction"
    PAUSE_REQUEST = "pause_request"


@dataclass
class RuntimeInput:
    kind: InputKind
    content: str = ""


class RuntimeInputChannel(Protocol):
    """Agent core 层依赖的最小接口."""

    def drain(self) -> list[RuntimeInput]: ...
