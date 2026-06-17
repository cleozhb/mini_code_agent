"""Mini Code Agent 入口 — 解析命令行参数，初始化组件，启动 REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from functools import partial
from pathlib import Path

from rich.console import Console

DEFAULT_TOKEN_BUDGET = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini Code Agent — 从零构建的编程 Agent",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "openai-responses", "anthropic"],
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
        "--prewarm-lsp",
        dest="prewarm_lsp",
        action="store_true",
        default=True,
        help="启动 REPL 前预热 Python LSP（默认开启）",
    )
    parser.add_argument(
        "--no-prewarm-lsp",
        dest="prewarm_lsp",
        action="store_false",
        help="不在启动 REPL 前预热 Python LSP，改为首次使用时按需启动",
    )

    # 子命令：不带 → REPL（保持原行为）；带 → 分派
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    from mini_code_agent.cli import add_eval_subparser
    add_eval_subparser(subparsers)

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

    # 子命令分派：eval 走 eval_cmd；其他（None）走 REPL
    if args.command == "eval":
        from mini_code_agent.cli import run_eval_command
        sys.exit(await run_eval_command(args))

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

    trace_recorder = None
    try:
        from mini_code_agent.trace import TraceRecorder

        trace_recorder = TraceRecorder(
            project_dir=project_dir,
            provider=args.provider,
            model=llm_client.model,
            on_error=lambda e: console.print(
                f"[yellow]trace 写入失败，主流程继续: {type(e).__name__}: {e}[/yellow]"
            ),
        )
        llm_client.set_trace_recorder(trace_recorder)
        console.print(
            f"[dim]trace: {trace_recorder.session_dir} "
            "(raw prompt / response / tool output)[/dim]"
        )
    except Exception as e:
        console.print(
            f"[yellow]trace 初始化失败，主流程继续: {type(e).__name__}: {e}[/yellow]"
        )

    # 2. 注册工具
    from mini_code_agent.tools import (
        AddMemoryTool,
        BashTool,
        EditFileTool,
        FindReferencesTool,
        GetDiagnosticsTool,
        GetHoverInfoTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
        GotoDefinitionTool,
        GrepTool,
        ListDirTool,
        LSPManager,
        ReadFileTool,
        RecallMemoryTool,
        SubAgentTool,
        ToolRegistry,
        WebFetchTool,
        WebSearchTool,
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

    # 创建 LSP 工具并注入 LSPManager
    lsp_manager = LSPManager()
    goto_def_tool = GotoDefinitionTool()
    goto_def_tool._lsp_manager = lsp_manager
    find_refs_tool = FindReferencesTool()
    find_refs_tool._lsp_manager = lsp_manager
    hover_tool = GetHoverInfoTool()
    hover_tool._lsp_manager = lsp_manager
    diagnostics_tool = GetDiagnosticsTool()
    diagnostics_tool._lsp_manager = lsp_manager

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())
    registry.register(add_memory_tool)
    registry.register(recall_memory_tool)
    registry.register(GitStatusTool())
    registry.register(GitDiffTool())
    registry.register(GitCommitTool())
    registry.register(GitLogTool())
    registry.register(goto_def_tool)
    registry.register(find_refs_tool)
    registry.register(hover_tool)
    registry.register(diagnostics_tool)
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(SubAgentTool(
        llm_client=llm_client,
        project_path=str(project_dir),
        lsp_manager=lsp_manager,
    ))

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
    from mini_code_agent.safety import CommandFilter, FileGuard, GitCheckpoint, LoopGuard

    command_filter = CommandFilter()
    file_guard = FileGuard(work_dir=project_dir)
    loop_guard = LoopGuard()

    # Git checkpoint：检查项目目录是否为 git 仓库
    git_checkpoint = GitCheckpoint(cwd=str(project_dir))
    if not await git_checkpoint.is_git_repo():
        console.print(
            "[yellow]⚠ 项目目录不是 git 仓库，自动 checkpoint 和 /undo 将不可用。"
            f"\n  如需启用，请先在 {project_dir} 执行 git init[/yellow]"
        )
        git_checkpoint = None

    # 5. 创建确认回调
    from prompt_toolkit import PromptSession
    from mini_code_agent.cli.confirm import confirm_tool_call
    from mini_code_agent.safety import SafetyLevel

    prompt_session = PromptSession()

    async def _confirm_cb(tool_name, tool_call, safety_level=SafetyLevel.NEEDS_CONFIRM):
        return await confirm_tool_call(
            tool_name, tool_call, console, prompt_session, safety_level,
        )

    # 6. 创建 Agent
    from mini_code_agent.core import Agent

    # Ledger + Checkpoint
    from mini_code_agent.longrun import (
        CheckpointManager,
        LongRunConfig,
        ResumeManager,
        TaskLedgerManager,
    )
    from mini_code_agent.artifacts import ArtifactStore
    from mini_code_agent.verify.verifier import IncrementalVerifier

    ledger_manager = TaskLedgerManager(
        storage_dir=str(project_dir / ".agent" / "ledger")
    )
    longrun_config = LongRunConfig()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(project_dir / ".agent" / "checkpoints"),
        ledger_manager=ledger_manager,
        git_checkpoint=git_checkpoint,
        cwd=str(project_dir),
    )

    artifact_store = ArtifactStore(
        storage_dir=str(project_dir / ".agent" / "artifacts"),
    )
    from mini_code_agent.verify.level1 import QuickVerifier

    incremental_verifier = IncrementalVerifier(
        level1=QuickVerifier(lsp_manager=lsp_manager),
    )

    if args.prewarm_lsp:
        try:
            await lsp_manager.start_server("python", str(project_dir))
            console.print("[dim]LSP: Python language server 已预热[/dim]")
        except FileNotFoundError as e:
            console.print(f"[yellow]LSP 预热跳过: {e}[/yellow]")
        except Exception as e:
            console.print(
                f"[yellow]LSP 预热失败，后续仍可按需启动: {type(e).__name__}: {e}[/yellow]"
            )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt,
        confirm_callback=_confirm_cb,
        command_filter=command_filter,
        file_guard=file_guard,
        loop_guard=loop_guard,
        project_memory=project_memory,
        git_checkpoint=git_checkpoint,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=longrun_config,
        incremental_verifier=incremental_verifier,
        trace_recorder=trace_recorder,
    )

    # 7. 启动 REPL
    from mini_code_agent.cli import REPL

    repl = REPL(
        agent=agent,
        console=console,
        artifact_store=artifact_store,
    )
    try:
        await repl.run()
    finally:
        if trace_recorder is not None:
            trace_recorder.finish_session()
        # 清理 LSP 服务器
        await lsp_manager.stop_server()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
