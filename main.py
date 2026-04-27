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
    parser.add_argument(
        "--graph",
        action="store_true",
        help="启动时默认开启 Graph 模式（DAG 任务图规划与执行）",
    )
    parser.add_argument(
        "--long-run",
        nargs="?",
        const=True,
        default=None,
        metavar="GOAL",
        help="启动长程任务模式。可直接传入目标，也可不带参数在 REPL 中交互输入",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"长程任务的 token 预算 (默认: {DEFAULT_TOKEN_BUDGET:,})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="TASK_ID",
        help="恢复之前的长程任务（传入 task_id 或 'list' 列出所有）",
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

    # 6b. 创建 GraphPlanner + GraphExecutor + Graph mode 回调
    from mini_code_agent.core import GraphPlanner, GraphExecutor
    from mini_code_agent.cli.graph_display import (
        ask_graph_blocked,
        render_task_progress,
    )

    graph_planner = GraphPlanner(llm_client=llm_client)

    async def _graph_blocked_cb(graph, failed, blocked):
        return await ask_graph_blocked(graph, failed, blocked, console, prompt_session)

    async def _graph_progress_cb(task_index, total, task, phase):
        render_task_progress(task_index, total, task, phase, console)

    # GraphExecutor 在下面拿到 SubtaskRunner / Ledger / Checkpoint 之后再实例化

    # 7. 创建 Agent
    from mini_code_agent.core import Agent

    # 7b. 长程任务 Ledger + Checkpoint + Resume
    from mini_code_agent.longrun import (
        CheckpointManager,
        LongRunConfig,
        ResumeManager,
        TaskLedgerManager,
    )

    ledger = None
    ledger_manager = TaskLedgerManager(
        storage_dir=str(project_dir / ".agent" / "ledger")
    )
    longrun_config = LongRunConfig(token_budget=args.token_budget)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(project_dir / ".agent" / "checkpoints"),
        ledger_manager=ledger_manager,
        git_checkpoint=git_checkpoint,
        cwd=str(project_dir),
    )
    resume_manager = ResumeManager(
        checkpoint_manager=checkpoint_manager,
        ledger_manager=ledger_manager,
        cwd=str(project_dir),
    )
    long_run_graph = None  # --long-run / --resume 生成的 TaskGraph

    if args.resume == "list":
        metas = ledger_manager.list_all()
        if not metas:
            console.print("[dim]没有可恢复的长程任务[/dim]")
            sys.exit(0)
        for m in metas:
            console.print(
                f"  {m.task_id[:8]}  [{m.status.value}]  "
                f"{m.goal[:60]}  "
                f"({m.completed_tasks} tasks, {m.total_tokens_used:,} tokens)"
            )
        sys.exit(0)
    elif args.resume:
        try:
            ledger = ledger_manager.load(args.resume)
            console.print(
                f"[green]已恢复长程任务: {ledger.task_id[:8]}[/green]\n"
                f"[dim]{ledger_manager.get_summary_for_resume(ledger)[:200]}...[/dim]"
            )
            # 从 ledger snapshot 重建 TaskGraph
            snapshot = ledger.task_graph_snapshot
            if snapshot and snapshot.get("nodes"):
                from mini_code_agent.core.task_graph import TaskGraph, TaskNode, TaskStatus
                long_run_graph = TaskGraph()
                long_run_graph.original_goal = snapshot.get("original_goal", ledger.goal)
                for _nid, ndata in snapshot["nodes"].items():
                    long_run_graph.add_task(TaskNode(
                        id=ndata["id"],
                        description=ndata["description"],
                        dependencies=ndata.get("dependencies", []),
                        status=TaskStatus(ndata.get("status", "pending")),
                        files_involved=ndata.get("files_involved", []),
                        verification=ndata.get("verification", ""),
                    ))
                # 把已完成的任务标记上
                for ct in ledger.completed_tasks:
                    if ct.task_id in long_run_graph.nodes:
                        node = long_run_graph.nodes[ct.task_id]
                        if node.status not in (TaskStatus.COMPLETED,):
                            long_run_graph.mark_completed(ct.task_id, ct.self_summary)
        except Exception as e:
            console.print(f"[red]恢复失败: {e}[/red]")
            sys.exit(1)
    elif args.long_run is True:
        # --long-run 不带目标：延迟到 REPL 交互式输入
        pass
    elif isinstance(args.long_run, str):
        # --long-run "目标"：立即生成 TaskGraph + 创建 Ledger
        console.print(f"[dim]正在为长程任务生成 TaskGraph...[/dim]")
        try:
            long_run_graph = await graph_planner.plan_as_graph(args.long_run)
        except Exception as e:
            console.print(f"[red]生成 TaskGraph 失败: {e}[/red]")
            sys.exit(1)

        from mini_code_agent.cli.graph_display import render_graph_table
        render_graph_table(long_run_graph, console)

        ledger = ledger_manager.create(
            goal=args.long_run,
            task_graph=long_run_graph,
            budget=args.token_budget,
        )
        console.print(
            f"[green]Ledger 已创建: {ledger.task_id[:8]}[/green]\n"
            f"[dim]Token 预算: {args.token_budget:,}  "
            f"里程碑: {len(ledger.milestones)} 个[/dim]"
        )

    # 7c. Artifact + Incremental Verifier — L1 集成所需
    from mini_code_agent.artifacts import ArtifactStore
    from mini_code_agent.verify.verifier import IncrementalVerifier
    from mini_code_agent.core import SubtaskRunner

    artifact_store = ArtifactStore(
        storage_dir=str(project_dir / ".agent" / "artifacts"),
    )
    incremental_verifier = IncrementalVerifier()

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
        plan_mode=args.plan,
        planner=planner,
        plan_confirm_callback=_plan_confirm_cb,
        plan_progress_callback=_plan_progress_cb,
        plan_replan_callback=_plan_replan_cb,
        ledger=ledger,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=longrun_config,
        task_graph=long_run_graph,
        incremental_verifier=incremental_verifier,
    )

    subtask_runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=incremental_verifier,
        git_checkpoint=git_checkpoint,
    )

    graph_executor = GraphExecutor(
        project_path=str(project_dir),
        blocked_callback=_graph_blocked_cb,
        progress_callback=_graph_progress_cb,
        subtask_runner=subtask_runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=longrun_config,
    )

    # 8. 启动 REPL
    from mini_code_agent.cli import REPL

    repl = REPL(
        agent=agent,
        console=console,
        graph_planner=graph_planner,
        graph_executor=graph_executor,
        graph_mode=args.graph,
        pending_graph=long_run_graph,
        long_run_deferred=args.long_run is True,
        token_budget=args.token_budget,
        checkpoint_manager=checkpoint_manager,
        resume_manager=resume_manager,
        artifact_store=artifact_store,
        longrun_config=longrun_config,
    )
    try:
        await repl.run()
    finally:
        # 清理 LSP 服务器
        await lsp_manager.stop_server()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
