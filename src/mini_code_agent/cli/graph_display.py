"""Task Graph 可视化 — 用 Rich 展示任务图的状态和进度."""

from __future__ import annotations

from typing import Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..core.task_graph import TaskGraph, TaskNode, TaskStatus

# ---------------------------------------------------------------------------
# 状态 emoji 映射
# ---------------------------------------------------------------------------

_STATUS_EMOJI: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "⬜",
    TaskStatus.READY: "🔵",
    TaskStatus.RUNNING: "🔄",
    TaskStatus.COMPLETED: "✅",
    TaskStatus.FAILED: "❌",
    TaskStatus.SKIPPED: "⏭️",
    TaskStatus.BLOCKED: "🚫",
}

_STATUS_STYLE: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "dim",
    TaskStatus.READY: "cyan",
    TaskStatus.RUNNING: "bold blue",
    TaskStatus.COMPLETED: "green",
    TaskStatus.FAILED: "red",
    TaskStatus.SKIPPED: "yellow",
    TaskStatus.BLOCKED: "dim red",
}


# ---------------------------------------------------------------------------
# 表格展示
# ---------------------------------------------------------------------------


def render_graph_table(graph: TaskGraph, console: Console) -> None:
    """用 Rich Table 渲染 Task Graph 状态."""
    if not graph.nodes:
        console.print("[dim]任务图为空[/dim]")
        return

    summary = graph.summary()
    total = len(graph)

    # 构建头部信息
    header_parts = [("Task Graph", "bold cyan")]
    if graph.original_goal:
        header_parts.extend([("\n目标：", "bold"), (graph.original_goal, "white")])
    header_parts.extend([
        ("\n", ""),
        (f"共 {total} 个任务", "dim"),
    ])
    for status_name, count in summary.items():
        header_parts.extend([("  ", ""), (f"{status_name}: {count}", "dim")])

    console.print(Panel(
        Text.assemble(*header_parts),
        border_style="cyan",
    ))

    # 任务表格
    table = Table(show_lines=True, expand=True)
    table.add_column("状态", width=4, justify="center")
    table.add_column("ID", style="bold", width=12)
    table.add_column("描述", ratio=3)
    table.add_column("依赖", ratio=1, overflow="fold")
    table.add_column("文件", style="magenta", ratio=2, overflow="fold")
    table.add_column("验证", style="dim", ratio=2, overflow="fold")

    for node in graph.nodes.values():
        emoji = _STATUS_EMOJI.get(node.status, "?")
        style = _STATUS_STYLE.get(node.status, "")

        deps_str = ", ".join(node.dependencies) if node.dependencies else "-"
        files_str = "\n".join(node.files_involved) if node.files_involved else "-"

        # 结果/错误信息
        desc = node.description
        if node.status == TaskStatus.COMPLETED and node.result:
            desc += f"\n[green dim]→ {node.result[:80]}[/green dim]"
        elif node.status == TaskStatus.FAILED and node.error:
            desc += f"\n[red dim]→ {node.error[:80]}[/red dim]"

        table.add_row(
            emoji,
            Text(node.id, style=style),
            desc,
            deps_str,
            files_str,
            node.verification or "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# 树形展示
# ---------------------------------------------------------------------------


def render_graph_tree(graph: TaskGraph, console: Console) -> None:
    """用 Rich Tree 渲染 Task Graph 的依赖树."""
    if not graph.nodes:
        console.print("[dim]任务图为空[/dim]")
        return

    title = f"[bold cyan]Task Graph[/bold cyan]"
    if graph.original_goal:
        title += f" — {graph.original_goal}"
    tree = Tree(title)

    # 找根节点（没有依赖的）
    roots = [n for n in graph.nodes.values() if not n.dependencies]
    if not roots:
        # 如果没有根节点（不应该发生），就显示所有节点
        roots = list(graph.nodes.values())

    visited: set[str] = set()

    def _add_node(parent: Tree, node: TaskNode) -> None:
        if node.id in visited:
            parent.add(f"[dim]↩ {node.id} (已显示)[/dim]")
            return
        visited.add(node.id)

        emoji = _STATUS_EMOJI.get(node.status, "?")
        style = _STATUS_STYLE.get(node.status, "")
        label = f"{emoji} [{style}]{node.id}[/{style}]: {node.description}"

        if node.status == TaskStatus.COMPLETED and node.result:
            label += f" [green dim]→ {node.result[:50]}[/green dim]"
        elif node.status == TaskStatus.FAILED and node.error:
            label += f" [red dim]→ {node.error[:50]}[/red dim]"

        branch = parent.add(label)

        # 找这个节点的下游
        children = [
            n for n in graph.nodes.values()
            if node.id in n.dependencies
        ]
        for child in children:
            _add_node(branch, child)

    for root in roots:
        _add_node(tree, root)

    console.print(tree)


# ---------------------------------------------------------------------------
# 进度渲染
# ---------------------------------------------------------------------------


def render_task_progress(
    task_index: int,
    total: int,
    task: TaskNode,
    phase: str,
    console: Console,
) -> None:
    """渲染单个任务的执行进度.

    phase: "start" | "end_ok" | "end_fail" | "retry"
    """
    if phase == "start":
        console.print()
        console.print(
            f"[bold cyan]▶ [{task_index}/{total}][/bold cyan] "
            f"[bold]正在执行：{task.description}[/bold]"
        )
        if task.files_involved:
            console.print(f"  [dim]文件：{', '.join(task.files_involved)}[/dim]")
        if task.verification:
            console.print(f"  [dim]验证：{task.verification}[/dim]")

    elif phase == "end_ok":
        console.print(
            f"[green]✓ [{task_index}/{total}] 完成：{task.description}[/green]"
        )

    elif phase == "end_fail":
        console.print(
            f"[red]✗ [{task_index}/{total}] 失败：{task.description}[/red]"
        )
        if task.error:
            console.print(f"  [red dim]{task.error[:200]}[/red dim]")

    elif phase == "retry":
        console.print(
            f"[yellow]↻ [{task_index}/{total}] 重试（第 {task.retry_count} 次）："
            f"{task.description}[/yellow]"
        )


# ---------------------------------------------------------------------------
# 阻塞时的用户交互
# ---------------------------------------------------------------------------


async def ask_graph_blocked(
    graph: TaskGraph,
    failed: list[TaskNode],
    blocked: list[TaskNode],
    console: Console,
    prompt_session: PromptSession,
) -> Literal["replan", "skip", "abort"]:
    """Task Graph 执行被阻塞时，询问用户下一步操作."""
    console.print()
    console.print("[bold red]Task Graph 执行被阻塞[/bold red]")
    console.print()

    if failed:
        console.print("[red]失败的任务：[/red]")
        for node in failed:
            console.print(f"  ❌ {node.id}: {node.description}")
            if node.error:
                console.print(f"     [dim]{node.error[:150]}[/dim]")

    if blocked:
        console.print("[dim]被阻塞的任务：[/dim]")
        for node in blocked:
            console.print(f"  🚫 {node.id}: {node.description}")

    console.print()
    console.print(
        "[bold]选择下一步：[/bold] "
        "[yellow]r[/yellow]=重新规划失败部分  "
        "[yellow]s[/yellow]=跳过失败任务继续  "
        "[yellow]a[/yellow]=放弃整个任务图"
    )

    try:
        ans = await prompt_session.prompt_async(HTML("<b>选择 [r/s/a]: </b>"))
    except (EOFError, KeyboardInterrupt):
        return "abort"

    choice = (ans or "").strip().lower()
    if choice in ("r", "replan"):
        return "replan"
    if choice in ("s", "skip"):
        return "skip"
    return "abort"


# ---------------------------------------------------------------------------
# Mermaid 导出
# ---------------------------------------------------------------------------


def render_mermaid(graph: TaskGraph, console: Console) -> None:
    """在终端展示 Mermaid 语法，方便复制到浏览器查看."""
    mermaid = graph.to_mermaid()
    console.print(Panel(
        mermaid,
        title="[bold]Mermaid 图表[/bold]",
        subtitle="[dim]复制到 mermaid.live 查看流程图[/dim]",
        border_style="cyan",
    ))


# ---------------------------------------------------------------------------
# 执行完成报告
# ---------------------------------------------------------------------------


def render_graph_result(
    result: "GraphResult",
    console: Console,
) -> None:
    """渲染 Task Graph 执行完成的汇总报告."""
    from ..core.graph_executor import GraphResult  # noqa: F811

    graph = result.graph
    lines = []

    if graph.is_complete():
        lines.append("[bold green]Task Graph 执行完成[/bold green]")
    elif graph.is_blocked():
        lines.append("[bold red]Task Graph 执行被阻塞[/bold red]")
    else:
        lines.append("[bold yellow]Task Graph 执行中断[/bold yellow]")

    lines.append("")
    lines.append(f"总步骤：{result.total_steps}")
    lines.append(f"完成：{result.tasks_completed}  失败：{result.tasks_failed}  跳过：{result.tasks_skipped}")
    lines.append(f"总 tokens：{result.total_tokens:,}")
    lines.append(f"墙钟时间：{result.wall_time:.1f}s")

    # 关键路径
    critical = graph.get_critical_path()
    if critical:
        lines.append("")
        lines.append(f"关键路径（{len(critical)} 步）：")
        path_str = " → ".join(n.id for n in critical)
        lines.append(f"  {path_str}")

    console.print(Panel(
        "\n".join(lines),
        border_style="cyan",
    ))
