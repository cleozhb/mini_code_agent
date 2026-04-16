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
        BashTool,
        GrepTool,
        ListDirTool,
        ReadFileTool,
        ToolRegistry,
        WriteFileTool,
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(BashTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())

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

    # 4. 创建确认回调
    from prompt_toolkit import PromptSession
    from mini_code_agent.cli.confirm import confirm_tool_call

    prompt_session = PromptSession()

    async def _confirm_cb(tool_name, tool_call):
        return await confirm_tool_call(tool_name, tool_call, console, prompt_session)

    # 5. 创建 Agent
    from mini_code_agent.core import Agent

    agent = Agent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt,
        confirm_callback=_confirm_cb,
    )

    # 6. 启动 REPL
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
