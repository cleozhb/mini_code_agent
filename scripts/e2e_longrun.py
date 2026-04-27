"""端到端测试驱动：跑一次完整的长程任务流程，不经过 REPL.

目的：
- 验证 GraphPlanner → SubtaskRunner → IncrementalVerifier → ArtifactStore → Ledger
  → CheckpointManager 在真实 LLM 调用下能闭环
- 同时演示 /artifacts 和 /resume（恢复）这两个能力
- 输出每个子任务的 confidence / verification 摘要、最终 /status 摘要
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

# 确保用 src 包，不串到已安装版本
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console

console = Console()


PROJECT_DIR = Path("/tmp/test-long-task")
GOAL = (
    "为这个空项目创建一个简单的 Todo 管理模块：\n"
    "1. src/todo/models.py 定义 Todo dataclass\n"
    "2. src/todo/storage.py 实现内存存储\n"
    "3. tests/test_todo.py 写测试，覆盖创建/读取/更新/删除"
)
TOKEN_BUDGET = 100_000


def _ensure_clean_project() -> None:
    """重建 /tmp/test-long-task 并 git init."""
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True)
    import subprocess
    sub = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_DIR, check=True, capture_output=True)
    sub("init", "-q", "-b", "main")
    sub("config", "user.email", "t@t")
    sub("config", "user.name", "t")
    (PROJECT_DIR / ".gitignore").write_text(".agent/\n__pycache__/\n", encoding="utf-8")
    sub("add", ".gitignore")
    sub("commit", "-q", "-m", "seed")


async def _build_runtime():
    """模仿 main.py，搭出一套完整运行时."""
    os.chdir(PROJECT_DIR)

    from mini_code_agent.llm import create_client
    from mini_code_agent.tools import (
        BashTool, EditFileTool, ReadFileTool, WriteFileTool,
        ListDirTool, GrepTool,
    )
    from mini_code_agent.tools.base import ToolRegistry
    from mini_code_agent.core import Agent, GraphExecutor, GraphPlanner, SubtaskRunner
    from mini_code_agent.core.system_prompt import build_system_prompt
    from mini_code_agent.safety.git_checkpoint import GitCheckpoint
    from mini_code_agent.artifacts import ArtifactStore
    from mini_code_agent.verify.verifier import IncrementalVerifier
    from mini_code_agent.longrun import (
        CheckpointManager, LongRunConfig, TaskLedgerManager,
    )

    llm_client = create_client(provider="openai", model=None)

    registry = ToolRegistry()
    for tool in (
        BashTool(), EditFileTool(), ReadFileTool(), WriteFileTool(),
        ListDirTool(), GrepTool(),
    ):
        registry.register(tool)

    git_checkpoint = GitCheckpoint(cwd=str(PROJECT_DIR))

    artifact_store = ArtifactStore(
        storage_dir=str(PROJECT_DIR / ".agent" / "artifacts"),
    )
    incremental_verifier = IncrementalVerifier()
    ledger_manager = TaskLedgerManager(
        storage_dir=str(PROJECT_DIR / ".agent" / "ledger"),
    )
    longrun_config = LongRunConfig(token_budget=TOKEN_BUDGET)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(PROJECT_DIR / ".agent" / "checkpoints"),
        ledger_manager=ledger_manager,
        git_checkpoint=git_checkpoint,
        cwd=str(PROJECT_DIR),
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=build_system_prompt(),
        project_path=str(PROJECT_DIR),
        git_checkpoint=git_checkpoint,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=longrun_config,
        incremental_verifier=incremental_verifier,
    )

    subtask_runner = SubtaskRunner(
        agent=agent,
        artifact_store=artifact_store,
        verifier=incremental_verifier,
        git_checkpoint=git_checkpoint,
    )

    executor = GraphExecutor(
        project_path=str(PROJECT_DIR),
        max_retries=1,  # 控制成本
        subtask_runner=subtask_runner,
        ledger_manager=ledger_manager,
        checkpoint_manager=checkpoint_manager,
        longrun_config=longrun_config,
    )

    graph_planner = GraphPlanner(llm_client=llm_client)

    return {
        "agent": agent,
        "executor": executor,
        "graph_planner": graph_planner,
        "ledger_manager": ledger_manager,
        "artifact_store": artifact_store,
        "checkpoint_manager": checkpoint_manager,
    }


def _print_artifact_event(artifact) -> None:
    """简洁打印一条 artifact 摘要 — 模拟 REPL 的实时反馈."""
    console.print(
        f"[bold cyan]│ {artifact.task_id}[/bold cyan]  "
        f"conf=[bold]{artifact.confidence.value}[/bold]  "
        f"files={artifact.patch.total_files_changed} "
        f"(+{artifact.patch.total_lines_added}/-{artifact.patch.total_lines_removed})  "
        f"verify={artifact.self_verification.summary()}  "
        f"scope={'✓' if artifact.scope_check.is_clean else '✗'}"
    )
    console.print(f"[dim]│   summary: {artifact.self_summary[:100]}[/dim]")


async def main() -> int:
    console.rule("[bold]步骤 0 — 初始化空项目[/bold]")
    _ensure_clean_project()
    console.print(f"[green]✓[/green] {PROJECT_DIR} 已重建（git init）")

    runtime = await _build_runtime()

    # 1. Plan
    console.rule("[bold]步骤 1 — 生成 TaskGraph[/bold]")
    console.print(f"[dim]目标：[/dim]\n{GOAL}\n")
    graph = await runtime["graph_planner"].plan_as_graph(GOAL)
    console.print(
        f"[green]✓[/green] 生成了 {len(graph)} 个子任务"
    )
    for nid, node in graph.nodes.items():
        deps = ",".join(node.dependencies) if node.dependencies else "—"
        console.print(
            f"  [{nid}] {node.description[:80]}  "
            f"deps=[{deps}]  files={node.files_involved}"
        )

    # 2. 创建 Ledger
    console.rule("[bold]步骤 2 — 创建 Ledger[/bold]")
    ledger = runtime["ledger_manager"].create(
        goal=GOAL, task_graph=graph, budget=TOKEN_BUDGET,
    )
    console.print(
        f"[green]✓[/green] Ledger {ledger.task_id[:8]}  "
        f"milestones={len(ledger.milestones)}  "
        f"budget={TOKEN_BUDGET:,}"
    )

    # 3. 把 ledger 注入 agent，让 system prompt 能拿到上下文
    runtime["agent"].ledger = ledger
    runtime["agent"].task_graph = graph

    # 4. 执行
    console.rule("[bold]步骤 3 — execute_with_ledger[/bold]")
    console.print("[dim]子任务执行实时摘要：[/dim]\n")

    # Hook：每个 artifact 落盘时打印一次（用 ArtifactStore.save 的 wrapper）
    artifact_store = runtime["artifact_store"]
    original_save = artifact_store.save

    def _wrapped_save(art):
        path = original_save(art)
        _print_artifact_event(art)
        return path

    artifact_store.save = _wrapped_save  # type: ignore[method-assign]

    try:
        result = await runtime["executor"].execute_with_ledger(
            graph, ledger, str(PROJECT_DIR),
        )
    except Exception as e:
        console.print(f"[red]execute_with_ledger 异常: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

    console.print(
        f"\n[bold]结果[/bold]  done={result.tasks_completed}  "
        f"failed={result.tasks_failed}  skipped={result.tasks_skipped}  "
        f"steps={result.total_steps}  "
        f"tokens={result.total_tokens:,}  "
        f"wall={result.wall_time:.1f}s"
    )

    # 5. /status 等价输出
    console.rule("[bold]步骤 4 — /status[/bold]")
    stats = runtime["ledger_manager"].get_stats(ledger)
    for k, v in stats.items():
        console.print(f"  {k}: {v}")

    # 6. /artifacts 等价输出
    console.rule("[bold]步骤 5 — /artifacts[/bold]")
    for ct in ledger.completed_tasks:
        metas = artifact_store.list_for_task(ct.task_id)
        console.print(f"[bold]{ct.task_id}[/bold]  ({len(metas)} artifact)")
        for m in metas:
            console.print(
                f"  {m.artifact_id[:8]}  conf={m.confidence}  "
                f"files={m.files_changed}  {m.created_at[:19]}"
            )

    if ledger.completed_tasks:
        any_id = artifact_store.list_for_task(
            ledger.completed_tasks[0].task_id
        )[0].artifact_id
        console.rule(f"[bold]步骤 5b — /artifacts show {any_id[:8]}[/bold]")
        art = artifact_store.load(any_id)
        console.print(art.summary_for_reviewer())
        console.print("\n[dim]--- diff preview ---[/dim]")
        console.print(art.diff_preview(max_lines=30))

    # 7. /resume 等价输出 — 重新加载 Ledger，验证持久化没丢
    console.rule("[bold]步骤 6 — /resume（重载 Ledger）[/bold]")
    reloaded = runtime["ledger_manager"].load(ledger.task_id)
    console.print(
        f"[green]✓[/green] 重新加载 Ledger {reloaded.task_id[:8]}\n"
        f"  goal: {reloaded.goal[:60]}...\n"
        f"  status: {reloaded.status.value}\n"
        f"  completed_tasks: {len(reloaded.completed_tasks)}\n"
        f"  failed_attempts: {len(reloaded.failed_attempts)}\n"
        f"  active_issues: {len(reloaded.active_issues)}\n"
        f"  total_tokens_used: {reloaded.total_tokens_used:,}"
    )

    # 8. 列出磁盘上的产物
    console.rule("[bold]步骤 7 — 磁盘产物[/bold]")
    import subprocess
    out = subprocess.run(
        ["find", ".agent", "-type", "f"], cwd=PROJECT_DIR,
        capture_output=True, text=True,
    ).stdout
    console.print(out or "(空)")

    # 检查项目实际生成的代码
    console.rule("[bold]步骤 8 — 项目最终文件树[/bold]")
    out = subprocess.run(
        ["find", ".", "-type", "f", "-not", "-path", "./.agent/*",
         "-not", "-path", "./.git/*"],
        cwd=PROJECT_DIR, capture_output=True, text=True,
    ).stdout
    console.print(out or "(空)")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
