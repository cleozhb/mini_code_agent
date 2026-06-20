"""TUI App — Textual full-screen 界面."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from time import monotonic
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Header, Input, RichLog, TextArea

from ..core.agent import Agent, AgentEvent, AgentEventType
from .command_handler import CommandHandler, CommandResult, RunnerCommand
from .repl import _changed_path_from_subagent_event
from .tui_confirm import ConfirmScreen
from .tui_input import PromptTextArea, TUIInputController
from .tui_renderer import TUIRenderer


class InputScreen(ModalScreen[str]):
    """单行输入弹窗，用于收集额外输入."""

    DEFAULT_CSS = """
    InputScreen {
        align: center middle;
    }
    #input-container {
        width: 60%;
        height: auto;
        max-height: 12;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #input-label {
        height: 1;
        margin-bottom: 1;
    }
    #input-field {
        height: 3;
    }
    """

    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        from textual.widgets import Static
        with Vertical(id="input-container"):
            yield Static(self._label, id="input-label")
            yield Input(placeholder="输入后按 Enter 确认，Escape 取消", id="input-field")

    def on_mount(self) -> None:
        self.query_one("#input-field", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        self.dismiss(text if text else "")

    def key_escape(self) -> None:
        self.dismiss("")


class TUIApp(App[None]):
    """Mini Code Agent TUI."""

    TITLE = "Mini Code Agent"
    BINDINGS = [Binding("ctrl+c", "quit", "退出", show=False)]

    CSS = """
    #output {
        height: 1fr;
        border: solid $accent;
    }
    #prompt-input {
        dock: bottom;
        height: 5;
        border: solid $primary;
    }
    """

    def __init__(self, agent: Agent, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._agent_busy = False
        self._input_ctrl: TUIInputController | None = None
        self._renderer: TUIRenderer | None = None
        self._cmd_handler: CommandHandler | None = None
        self._pending_goal: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="output", wrap=True, highlight=True)
        yield PromptTextArea(id="prompt-input")

    def on_mount(self) -> None:
        output = self.query_one("#output", RichLog)
        self._renderer = TUIRenderer(output)
        self._cmd_handler = CommandHandler(self._agent, self._renderer)
        self.sub_title = self._agent.llm_client.model
        self.query_one("#prompt-input", PromptTextArea).focus()

    @on(PromptTextArea.Submitted)
    async def handle_prompt_submitted(self, event: PromptTextArea.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return

        self._renderer.render_user_input(text)

        if self._agent_busy:
            if self._input_ctrl:
                self._input_ctrl.put(text)
                self._renderer.render_system(f"已发送: {text}")
        elif text.startswith("/"):
            result = await self._cmd_handler.handle(text)
            if result == CommandResult.QUIT:
                self.exit()
            elif isinstance(result, RunnerCommand):
                await self._dispatch_runner(result)
            elif result == CommandResult.UNKNOWN:
                self._renderer.render_error(f"未知命令: {text}")
        else:
            self._run_agent(text)

    async def _dispatch_runner(self, cmd: RunnerCommand) -> None:
        if cmd.kind == "goal":
            self._start_goal(cmd.arg)
        elif cmd.kind == "plan":
            self._run_plan(cmd.arg)
        elif cmd.kind == "exec":
            self._run_exec(cmd.arg)
        else:
            self._renderer.render_error(f"TUI 暂不支持: /{cmd.kind}")

    def _start_goal(self, arg: str) -> None:
        goal = arg.strip()
        if not goal:
            self._renderer.render_error("用法: /goal <目标描述>")
            return
        if goal == "resume" or goal.startswith("resume "):
            parts = goal.split(maxsplit=1)
            self._do_goal_resume(parts[1].strip() if len(parts) > 1 else None)
            return
        self._pending_goal = goal
        self.push_screen(
            InputScreen("请输入成功标准（如何判定目标已达成）："),
            callback=self._on_goal_criteria,
        )

    def _on_goal_criteria(self, result: str) -> None:
        goal = self._pending_goal
        self._pending_goal = None
        criteria = result.strip() if result else ""
        if not criteria:
            self._renderer.render_error("标准为空，已取消")
            return
        self._do_run_goal(goal, criteria)

    @work(exclusive=True, exit_on_error=False)
    async def _run_agent(self, user_input: str) -> None:
        self._agent_busy = True
        self._input_ctrl = TUIInputController()
        try:
            prompt = user_input
            while True:
                async for event in self._agent.run_stream(prompt, input_channel=self._input_ctrl):
                    self._renderer.dispatch_event(event)
                # run_stream 结束后，检查是否有未消费的用户输入
                remaining = self._input_ctrl.drain()
                from ..core.runtime_input import InputKind
                user_msgs = [i.content for i in remaining if i.kind == InputKind.USER_INSTRUCTION and i.content]
                if user_msgs:
                    prompt = "\n".join(user_msgs)
                else:
                    break
        except Exception as e:
            self._renderer.render_error(f"Agent error: {e}")
        finally:
            self._agent_busy = False

    # ------------------------------------------------------------------
    # /goal
    # ------------------------------------------------------------------

    @work(exclusive=True, exit_on_error=False)
    async def _do_run_goal(self, goal: str, criteria: str) -> None:
        from ..core.goal_prompt import build_goal_driven_prompt
        from ..longrun.checkpoint_manager import CheckpointManager
        from ..longrun.config import LongRunConfig
        from ..safety.loop_guard import LoopGuard
        from ..tools.base import ToolRegistry
        from ..tools.file_ops import ReadFileTool
        from ..tools.git import GitLogTool, GitStatusTool
        from ..tools.search import GrepTool, ListDirTool
        from ..tools.shell import BashTool
        from ..tools.subagent import CODER_SUBAGENT_PROMPT, SubAgentTool
        project_path = str(Path(".").resolve())
        llm_client = self._agent.llm_client
        self._agent_busy = True
        self._input_ctrl = TUIInputController()
        subagent_files_changed: list[str] = []

        async def _relay_subagent_event(event: AgentEvent) -> None:
            changed_path = _changed_path_from_subagent_event(event)
            if changed_path and changed_path not in subagent_files_changed:
                subagent_files_changed.append(changed_path)
            await self._relay_subagent_event(event)

        try:
            sub_agent_tool = SubAgentTool(
                llm_client=llm_client,
                project_path=project_path,
                system_prompt=CODER_SUBAGENT_PROMPT,
                confirm_callback=self._agent.confirm_callback,
                event_callback=_relay_subagent_event,
                lsp_manager=getattr(self._agent, "lsp_manager", None),
                input_channel=self._input_ctrl,
            )

            registry = ToolRegistry()
            registry.register(sub_agent_tool)
            registry.register(BashTool(cwd=project_path))
            registry.register(ReadFileTool())
            registry.register(GrepTool())
            registry.register(ListDirTool())
            registry.register(GitStatusTool())
            registry.register(GitLogTool())

            system_prompt = build_goal_driven_prompt(goal, criteria, project_path)

            longrun_config = LongRunConfig()
            checkpoint_manager = None
            manager = self._agent.ledger_manager
            if manager is not None and self._agent.git_checkpoint is not None:
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=str(Path(project_path) / ".agent" / "checkpoints"),
                    ledger_manager=manager,
                    git_checkpoint=self._agent.git_checkpoint,
                    cwd=project_path,
                )

            master_agent = Agent(
                llm_client=llm_client,
                tool_registry=registry,
                system_prompt=system_prompt,
                confirm_callback=self._agent.confirm_callback,
                loop_guard=LoopGuard(max_rounds=200),
                project_path=project_path,
                git_checkpoint=self._agent.git_checkpoint,
                checkpoint_manager=checkpoint_manager,
                longrun_config=longrun_config,
            )

            ledger = None
            if manager is not None:
                from ..longrun.current_goal import apply_trace_context, save_current_goal
                from ..longrun.goal_observer import attach_goal_ledger_observer
                from ..longrun.ledger_types import TaskRunStatus
                ledger = manager.create(goal=goal, budget=500_000)
                apply_trace_context(ledger, self._agent.trace_recorder)
                ledger.status = TaskRunStatus.RUNNING
                ledger.current_phase = "execution"
                ledger.current_task_id = "goal"
                ledger.task_graph_snapshot.setdefault("nodes", {})["goal"] = {
                    "id": "goal", "description": goal, "status": "running",
                }
                ledger.task_graph_snapshot["criteria"] = criteria
                ledger.token_budget_remaining = max(0, ledger.token_budget - ledger.total_tokens_used)
                manager.save(ledger)
                save_current_goal(project_path, ledger)
                master_agent.ledger = ledger
                master_agent.ledger_manager = manager
                attach_goal_ledger_observer(master_agent, ledger, manager, subagent_files_changed)
                self._renderer.render_system(f"Ledger 已创建: {ledger.task_id[:8]}")

            self._renderer.render_system(f"Goal-Driven 模式启动 | 目标: {goal}")

            first_prompt = f"开始执行。当前项目根目录是 {project_path}。先用相对路径检查当前项目状态，然后决定第一步。"
            await self._goal_loop(master_agent, ledger, manager, checkpoint_manager, longrun_config, first_prompt)
        except Exception as e:
            self._renderer.render_error(f"Goal error: {e}")
        finally:
            self._agent_busy = False

    async def _goal_loop(self, master_agent, ledger, manager, checkpoint_manager, longrun_config, initial_prompt: str) -> None:
        goal_status = "active"
        run_interrupted = False
        run_error: Exception | None = None
        started_at = monotonic()

        async def run_one_turn(prompt: str) -> str:
            nonlocal run_interrupted, run_error
            text_acc = ""
            try:
                async for event in master_agent.run_stream(prompt, input_channel=self._input_ctrl):
                    self._renderer.dispatch_event(event)
                    if event.type == AgentEventType.TEXT_DELTA:
                        text_acc += event.content
                    elif event.type == AgentEventType.INTERRUPTED:
                        run_interrupted = True
            except (KeyboardInterrupt, asyncio.CancelledError):
                run_interrupted = True
            except Exception as e:
                run_error = e
                self._renderer.render_error(f"错误: {type(e).__name__}: {e}")
            return text_acc

        def parse_status(text: str) -> str:
            matches = list(re.finditer(
                r"^\s*\[?\s*goal_status\s*:\s*(active|complete|blocked)\b\s*\]?\s*$",
                text, flags=re.IGNORECASE | re.MULTILINE,
            ))
            return matches[-1].group(1).lower() if matches else "active"

        output = await run_one_turn(initial_prompt)
        goal_status = parse_status(output)

        while goal_status == "active" and not run_interrupted and run_error is None:
            next_prompt = "继续执行下一步。"
            if self._input_ctrl:
                from ..core.runtime_input import InputKind
                items = self._input_ctrl.drain()
                if any(i.kind == InputKind.PAUSE_REQUEST for i in items):
                    run_interrupted = True
                    break
                user_items = [i.content for i in items if i.kind == InputKind.USER_INSTRUCTION and i.content]
                if user_items:
                    next_prompt = "\n".join(user_items)
            output = await run_one_turn(next_prompt)
            if run_interrupted or run_error is not None:
                break
            goal_status = parse_status(output)

        if manager is not None and ledger is not None:
            from ..longrun.ledger_types import TaskRunStatus
            ledger.total_wall_time_seconds += monotonic() - started_at
            if run_interrupted:
                ledger.status = TaskRunStatus.PAUSED
                ledger.current_phase = "paused"
            elif run_error is not None:
                ledger.status = TaskRunStatus.FAILED
                ledger.current_phase = "failed"
            elif goal_status == "complete":
                ledger.status = TaskRunStatus.COMPLETED
                ledger.current_phase = "done"
            elif goal_status == "blocked":
                ledger.status = TaskRunStatus.PAUSED
                ledger.current_phase = "blocked"
            else:
                ledger.status = TaskRunStatus.FAILED
                ledger.current_phase = "failed"
            ledger.token_budget_remaining = max(0, ledger.token_budget - ledger.total_tokens_used)
            manager.save(ledger)
            from ..longrun.current_goal import clear_current_goal, save_current_goal
            if ledger.status in (TaskRunStatus.PAUSED, TaskRunStatus.RUNNING):
                save_current_goal(master_agent.project_path or str(Path(".").resolve()), ledger)
            else:
                clear_current_goal(master_agent.project_path or str(Path(".").resolve()), ledger.task_id)

        if run_interrupted and checkpoint_manager is not None and ledger is not None:
            from ..longrun.session_state import CheckpointTrigger
            try:
                from ..longrun.message_checkpoint import serialize_checkpoint_messages
                msg_dicts = serialize_checkpoint_messages(master_agent.messages)
                state = await checkpoint_manager.save_checkpoint(
                    ledger=ledger, trigger=CheckpointTrigger.USER_PAUSE,
                    config=longrun_config, current_task_id="goal",
                    recent_messages=msg_dicts,
                )
                ledger.last_checkpoint_id = state.checkpoint_id
                manager.save(ledger)
                from ..longrun.current_goal import save_current_goal
                save_current_goal(master_agent.project_path or str(Path(".").resolve()), ledger)
                self._renderer.render_system("Checkpoint 已保存")
            except Exception:
                pass

        self._renderer.render_system("Goal-Driven 模式结束")

    # ------------------------------------------------------------------
    # /goal resume
    # ------------------------------------------------------------------

    @work(exclusive=True, exit_on_error=False)
    async def _do_goal_resume(self, task_id_prefix: str | None = None) -> None:
        from ..longrun.checkpoint_manager import CheckpointManager
        from ..longrun.config import LongRunConfig
        from ..longrun.ledger_types import TaskRunStatus
        from ..safety.loop_guard import LoopGuard
        from ..tools.base import ToolRegistry
        from ..tools.file_ops import ReadFileTool
        from ..tools.git import GitLogTool, GitStatusTool
        from ..tools.search import GrepTool, ListDirTool
        from ..tools.shell import BashTool
        from ..tools.subagent import CODER_SUBAGENT_PROMPT, SubAgentTool

        manager = self._agent.ledger_manager
        if manager is None:
            self._renderer.render_error("Ledger 未启用")
            return

        project_path = str(Path(".").resolve())
        try:
            from ..longrun.current_goal import (
                format_goal_candidates,
                list_resumable_goals,
                resolve_goal_to_resume,
            )
            from ..longrun.ledger_manager import LedgerError
            ledger = resolve_goal_to_resume(manager, project_path, task_id_prefix)
        except LedgerError as e:
            from ..longrun.current_goal import format_goal_candidates, list_resumable_goals
            candidates = list_resumable_goals(manager)
            if candidates:
                self._renderer.render_error(
                    f"无法确定要恢复的 Goal: {e}\n"
                    "请使用 /goal resume <task_id> 指定：\n"
                    f"{format_goal_candidates(candidates)}"
                )
            else:
                self._renderer.render_system("没有可恢复的 Goal")
            return
        goal = ledger.task_graph_snapshot.get("original_goal", "") or ledger.goal
        criteria = ledger.task_graph_snapshot.get("criteria", "")
        if not goal:
            self._renderer.render_error("无法恢复：Goal 信息缺失")
            return

        project_path = str(Path(".").resolve())
        llm_client = self._agent.llm_client
        self._agent_busy = True
        self._input_ctrl = TUIInputController()
        subagent_files_changed: list[str] = []

        async def _relay_subagent_event(event: AgentEvent) -> None:
            changed_path = _changed_path_from_subagent_event(event)
            if changed_path and changed_path not in subagent_files_changed:
                subagent_files_changed.append(changed_path)
            await self._relay_subagent_event(event)

        try:
            # 加载 checkpoint
            checkpoint_manager = None
            restore_messages = None
            if self._agent.git_checkpoint is not None:
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=str(Path(project_path) / ".agent" / "checkpoints"),
                    ledger_manager=manager,
                    git_checkpoint=self._agent.git_checkpoint,
                    cwd=project_path,
                )
                checkpoint_id = ledger.last_checkpoint_id
                if checkpoint_id is None:
                    checkpoints = checkpoint_manager.list_checkpoints(ledger.task_id)
                    checkpoint_id = checkpoints[0].id if checkpoints else None
                if checkpoint_id:
                    try:
                        from ..longrun.message_checkpoint import restore_checkpoint_messages
                        state = checkpoint_manager.load_checkpoint(ledger.task_id, checkpoint_id)
                        restore_messages = restore_checkpoint_messages(state.recent_messages_full or [])
                        self._renderer.render_system(f"已加载 checkpoint {checkpoint_id[:8]}")
                    except Exception:
                        pass

            # 构建恢复 prompt
            resume_summary = manager.get_summary_for_resume(ledger)
            if criteria:
                initial_prompt = (
                    f"{resume_summary}\n\n"
                    f"成功标准: {criteria}\n\n"
                    f"请从中断处继续执行。不需要重做已完成的工作。"
                )
            else:
                initial_prompt = (
                    f"{resume_summary}\n\n"
                    f"请从中断处继续执行。不需要重做已完成的工作。"
                )

            # 构建 master agent
            from ..core.agent import Agent
            from ..core.goal_prompt import build_goal_driven_prompt

            sub_agent_tool = SubAgentTool(
                llm_client=llm_client,
                project_path=project_path,
                system_prompt=CODER_SUBAGENT_PROMPT,
                confirm_callback=self._agent.confirm_callback,
                event_callback=_relay_subagent_event,
                lsp_manager=getattr(self._agent, "lsp_manager", None),
                input_channel=self._input_ctrl,
            )
            registry = ToolRegistry()
            registry.register(sub_agent_tool)
            registry.register(BashTool(cwd=project_path))
            registry.register(ReadFileTool())
            registry.register(GrepTool())
            registry.register(ListDirTool())
            registry.register(GitStatusTool())
            registry.register(GitLogTool())

            system_prompt = build_goal_driven_prompt(goal, criteria, project_path)
            longrun_config = LongRunConfig()

            master_agent = Agent(
                llm_client=llm_client,
                tool_registry=registry,
                system_prompt=system_prompt,
                confirm_callback=self._agent.confirm_callback,
                loop_guard=LoopGuard(max_rounds=200),
                project_path=project_path,
                git_checkpoint=self._agent.git_checkpoint,
                checkpoint_manager=checkpoint_manager,
                longrun_config=longrun_config,
            )

            from ..longrun.current_goal import apply_trace_context, save_current_goal
            apply_trace_context(ledger, self._agent.trace_recorder)
            ledger.status = TaskRunStatus.RUNNING
            ledger.current_phase = "execution"
            master_agent.ledger = ledger
            master_agent.ledger_manager = manager
            from ..longrun.goal_observer import attach_goal_ledger_observer
            attach_goal_ledger_observer(master_agent, ledger, manager, subagent_files_changed)
            manager.save(ledger)
            save_current_goal(project_path, ledger)

            # 注入恢复对话历史
            if restore_messages:
                for msg in restore_messages:
                    master_agent.conversation.append(msg)

            self._renderer.render_system(f"恢复 Goal: {goal}")
            await self._goal_loop(master_agent, ledger, manager, checkpoint_manager, longrun_config, initial_prompt)
        except Exception as e:
            self._renderer.render_error(f"Goal resume error: {e}")
        finally:
            self._agent_busy = False

    @work(exclusive=True, exit_on_error=False)
    async def _run_plan(self, arg: str) -> None:
        from ..tools.subagent import SubAgentTool

        goal = arg.strip()
        if not goal:
            self._renderer.render_error("用法: /plan <需求描述>")
            return

        self._agent_busy = True
        self._input_ctrl = TUIInputController()
        project_path = str(Path(".").resolve())

        try:
            sub_tool = SubAgentTool(
                llm_client=self._agent.llm_client,
                project_path=project_path,
                confirm_callback=self._agent.confirm_callback,
                event_callback=self._relay_subagent_event,
                lsp_manager=getattr(self._agent, "lsp_manager", None),
            )
            self._renderer.render_system(f"正在规划: {goal}")
            result = await sub_tool.execute(goal=goal, context="", type="plan")
            if result.error:
                self._renderer.render_error(result.error)
            else:
                from rich.markdown import Markdown
                self._renderer._output.write(Markdown(result.output))
        except Exception as e:
            self._renderer.render_error(f"Plan error: {e}")
        finally:
            self._agent_busy = False

    # ------------------------------------------------------------------
    # /exec
    # ------------------------------------------------------------------

    @work(exclusive=True, exit_on_error=False)
    async def _run_exec(self, arg: str) -> None:
        file_path = arg.strip()
        if not file_path:
            self._renderer.render_error("用法: /exec <方案文件路径>")
            return

        target = Path(file_path)
        if not target.is_file():
            self._renderer.render_error(f"文件不存在: {file_path}")
            return

        content = target.read_text(encoding="utf-8")
        prompt = f"请按照以下方案文件执行实施：\n\n---\n{content}\n---\n\n逐步执行方案中的步骤，完成后简要总结。"

        self._agent_busy = True
        self._input_ctrl = TUIInputController()
        try:
            async for event in self._agent.run_stream(prompt, input_channel=self._input_ctrl):
                self._renderer.dispatch_event(event)
        except Exception as e:
            self._renderer.render_error(f"Exec error: {e}")
        finally:
            self._agent_busy = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _relay_subagent_event(self, event: AgentEvent) -> None:
        self._renderer.render_subagent_event(event)

    async def _tui_confirm_cb(
        self, tool_name: str, tool_call: Any, safety_level: Any
    ) -> tuple[bool, dict[str, Any] | None]:
        future: asyncio.Future[tuple[bool, dict[str, Any] | None]] = asyncio.get_event_loop().create_future()

        def on_dismiss(result: tuple[bool, dict[str, Any] | None]) -> None:
            if not future.done():
                future.set_result(result)

        self.push_screen(
            ConfirmScreen(tool_name, tool_call, safety_level),
            callback=on_dismiss,
        )
        return await future
