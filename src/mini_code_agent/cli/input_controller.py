"""运行时输入控制器 — 管理后台输入收集，实现 RuntimeInputChannel Protocol.

使用非阻塞 stdin 读取代替 prompt_toolkit，避免与 Rich 输出冲突。
"""

from __future__ import annotations

import asyncio
import sys
import termios
import tty

from ..core.runtime_input import InputKind, RuntimeInput, RuntimeInputChannel


class RuntimeInputController:
    """管理后台输入收集。实现 RuntimeInputChannel Protocol.

    通过 asyncio reader 监听 stdin，按 Enter 提交一行。
    不使用 prompt_toolkit，因此不与 Rich console 输出冲突。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[RuntimeInput] = asyncio.Queue()
        self._bg_task: asyncio.Task | None = None
        self._old_termios: list | None = None

    def start(self) -> None:
        if self._bg_task and not self._bg_task.done():
            return
        self._bg_task = asyncio.create_task(self._input_loop())

    async def stop(self) -> None:
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
        self._bg_task = None
        self._restore_terminal()

    async def suspend(self) -> None:
        await self.stop()

    def resume(self) -> None:
        self.start()

    def put(self, text: str) -> None:
        stripped = text.strip()
        if stripped in ("/pause", "/goal pause"):
            self._queue.put_nowait(RuntimeInput(kind=InputKind.PAUSE_REQUEST))
        elif stripped.startswith("/"):
            sys.stdout.write(f"\n⚠ 运行中不支持 {stripped}，请等本轮结束后输入\n")
            sys.stdout.flush()
        else:
            self._queue.put_nowait(RuntimeInput(kind=InputKind.USER_INSTRUCTION, content=text))

    def drain(self) -> list[RuntimeInput]:
        items: list[RuntimeInput] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def _restore_terminal(self) -> None:
        if self._old_termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError):
                pass
            self._old_termios = None

    async def _input_loop(self) -> None:
        """用 asyncio 的 reader 回调监听 stdin，逐行收集输入."""
        loop = asyncio.get_running_loop()
        line_buf: list[str] = []

        try:
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (termios.error, OSError, ValueError):
            return

        event = asyncio.Event()
        buf_container: list[list[str]] = [line_buf]

        def _on_stdin_readable() -> None:
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                return
            if not ch:
                return
            if ch in ("\n", "\r"):
                event.set()
            elif ch == "\x7f" or ch == "\x08":  # backspace
                if buf_container[0]:
                    buf_container[0].pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x03":  # Ctrl+C — 不吞，让外层捕获
                self._restore_terminal()
                raise KeyboardInterrupt
            elif ch >= " ":
                buf_container[0].append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()

        try:
            loop.add_reader(fd, _on_stdin_readable)
            while True:
                await event.wait()
                event.clear()
                text = "".join(buf_container[0])
                buf_container[0] = []
                sys.stdout.write("\n")
                sys.stdout.flush()
                if text.strip():
                    self.put(text)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                loop.remove_reader(fd)
            except (OSError, ValueError):
                pass
            self._restore_terminal()

