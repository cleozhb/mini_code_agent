"""Mini Code Agent 入口 — 解析命令行参数，初始化组件，启动 REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from functools import partial
from pathlib import Path

from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini Code Agent — 从零构建的编程 Agent",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM 服务商 (默认: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称，不指定则从 .env 读取",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="项目根目录 (默认: 当前目录)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息（上下文统计等）",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="启动时默认开启 Plan 模式（先规划再执行）",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    console = Console()

    # 切换到项目目录
    project_dir = Path(args.project_dir).resolve()
    if not project_dir.is_dir():
        console.print(f"[red]项目目录不存在: {project_dir}[/red]")
        sys.exit(1)
    os.chdir(project_dir)

    # 1. 创建 LLM 客户端
    from mini_code_agent.llm import create_client

    try:
        llm_client = create_client(
            provider=args.provider,
            model=args.model,
        )
    except Exception as e:
        console.print(f"[red]创建 LLM 客户端失败: {e}[/red]")
        console.print("[dim]请检查 .env 文件中的 API Key 配置[/dim]")
        sys.exit(1)

    # 2. 注册工具
    from mini_code_agent.tools import (
        AddMemoryTool,
        BashTool,
        EditFileTool,
        GrepTool,
        ListDirTool,
        ReadFileTool,
        RecallMemoryTool,
        ToolRegistry,
        WriteFileTool,
    )
    from mini_code_agent.memory import ProjectMemory

    # 初始化项目记忆
    project_memory = ProjectMemory(project_dir)

    # 创建记忆工具并注入 ProjectMemory
    add_memory_tool = AddMemoryTool()
    add_memory_tool._project_memory = project_memory
    recall_memory_tool = RecallMemoryTool()
    recall_memory_tool._project_memory = project_memory

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())
    registry.register(add_memory_tool)
    registry.register(recall_memory_tool)

    # 3. 构建 system prompt（使用项目上下文感知）
    from mini_code_agent.core import build_system_prompt_with_context
    from mini_code_agent.context import ContextBudget

    budget = ContextBudget()
    system_prompt, context_builder = build_system_prompt_with_context(
        project_path=project_dir,
        budget=budget,
    )

    # 打印上下文统计
    if args.verbose:
        stats = context_builder.get_context_stats()
        info = context_builder.project_info
        console.print(
            f"[dim][Context] "
            f"project: {info.name or project_dir.name} ({info.language}) | "
            f"initial: {stats.initial_context_tokens:,} tokens | "
            f"budget remaining: {stats.remaining_tokens:,} tokens"
            f"[/dim]"
        )

    # 4. 创建安全控制层
    from mini_code_agent.safety import CommandFilter, FileGuard, LoopGuard

    command_filter = CommandFilter()
    file_guard = FileGuard(work_dir=project_dir)
    loop_guard = LoopGuard()

    # 5. 创建确认回调
    from prompt_toolkit import PromptSession
    from mini_code_agent.cli.confirm import confirm_tool_call
    from mini_code_agent.safety import SafetyLevel

    prompt_session = PromptSession()

    async def _confirm_cb(tool_name, tool_call, safety_level=SafetyLevel.NEEDS_CONFIRM):
        return await confirm_tool_call(
            tool_name, tool_call, console, prompt_session, safety_level,
        )

    # 6. 创建 Planner + Plan mode 回调
    from mini_code_agent.core import Planner
    from mini_code_agent.cli.plan_display import (
        ask_replan,
        confirm_plan,
        render_step_done,
        render_step_start,
    )

    planner = Planner(llm_client=llm_client)

    async def _plan_confirm_cb(plan):
        result = await confirm_plan(plan, console, prompt_session)
        return (result.decision == "confirm", result.plan)

    async def _plan_progress_cb(idx, total, step, phase, success):
        if phase == "start":
            render_step_start(idx, total, step, console)
        else:
            render_step_done(idx, total, step, success, console)

    async def _plan_replan_cb(plan, failed_step, last_content):
        return await ask_replan(console, prompt_session)

    # 7. 创建 Agent
    from mini_code_agent.core import Agent

    agent = Agent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt,
        confirm_callback=_confirm_cb,
        command_filter=command_filter,
        file_guard=file_guard,
        loop_guard=loop_guard,
        project_memory=project_memory,
        plan_mode=args.plan,
        planner=planner,
        plan_confirm_callback=_plan_confirm_cb,
        plan_progress_callback=_plan_progress_cb,
        plan_replan_callback=_plan_replan_cb,
    )

    # 8. 启动 REPL
    from mini_code_agent.cli import REPL

    repl = REPL(agent=agent, console=console)
    await repl.run()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
