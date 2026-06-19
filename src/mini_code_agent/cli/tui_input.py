"""TUI 输入组件 — PromptTextArea + TUIInputController + 命令补全."""

from __future__ import annotations

import asyncio

from textual.message import Message
from textual.widgets import TextArea, OptionList
from textual.widgets.option_list import Option

from ..core.runtime_input import InputKind, RuntimeInput, RuntimeInputChannel

_COMMANDS = [
    "/goal", "/plan", "/exec",
    "/clear", "/cost", "/model",
    "/memory", "/save",
    "/undo", "/checkpoints", "/diff",
    "/quit", "/exit", "/q",
    "/help",
]


class PromptTextArea(TextArea):
    """多行输入框：Enter 提交，Esc+Enter 换行，/ 触发命令补全."""

    class Submitted(Message):
        def __init__(self, text_area: PromptTextArea, text: str) -> None:
            self.text_area = text_area
            self.text = text
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(language=None, **kwargs)
        self._completions: OptionList | None = None

    async def _on_key(self, event) -> None:
        if self._completions is not None and self._completions.display:
            if event.key == "down":
                event.stop()
                event.prevent_default()
                self._completions.action_cursor_down()
                return
            elif event.key == "up":
                event.stop()
                event.prevent_default()
                self._completions.action_cursor_up()
                return
            elif event.key in ("enter", "tab"):
                event.stop()
                event.prevent_default()
                idx = self._completions.highlighted
                if idx is not None:
                    option = self._completions.get_option_at_index(idx)
                    self.clear()
                    self.insert(option.prompt)
                self._hide_completions()
                return
            elif event.key == "escape":
                event.stop()
                event.prevent_default()
                self._hide_completions()
                return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(self, text))
            self.clear()
            self._hide_completions()
            return
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            self._do_tab_complete()
            return
        await super()._on_key(event)
        self._update_completions()

    def _update_completions(self) -> None:
        text = self.text
        if text.startswith("/") and " " not in text:
            matches = [c for c in _COMMANDS if c.startswith(text)]
            if matches and text != matches[0]:
                self._show_completions(matches)
            else:
                self._hide_completions()
        else:
            self._hide_completions()

    def _show_completions(self, items: list[str]) -> None:
        if self._completions is None:
            self._completions = OptionList(id="cmd-completions")
            self._completions.styles.dock = "bottom"
            self._completions.styles.height = "auto"
            self._completions.styles.max_height = "6"
            self._completions.styles.layer = "overlay"
            try:
                self.screen.mount(self._completions, before=self)
            except Exception:
                self._completions = None
                return
        self._completions.clear_options()
        for item in items:
            self._completions.add_option(Option(item))
        self._completions.display = True

    def _hide_completions(self) -> None:
        if self._completions is not None:
            self._completions.display = False

    def _do_tab_complete(self) -> None:
        text = self.text
        if text.startswith("/") and " " not in text:
            matches = [c for c in _COMMANDS if c.startswith(text)]
            if len(matches) == 1:
                self.clear()
                self.insert(matches[0] + " ")
                self._hide_completions()
            elif matches:
                self._show_completions(matches)
        else:
            self._hide_completions()


class TUIInputController(RuntimeInputChannel):
    """通过 asyncio.Queue 接收用户运行时输入."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[RuntimeInput] = asyncio.Queue()

    def put(self, text: str) -> None:
        text = text.strip()
        if text == "/pause":
            self._queue.put_nowait(RuntimeInput(kind=InputKind.PAUSE_REQUEST))
        else:
            self._queue.put_nowait(RuntimeInput(kind=InputKind.USER_INSTRUCTION, content=text))

    def drain(self) -> list[RuntimeInput]:
        items: list[RuntimeInput] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items
