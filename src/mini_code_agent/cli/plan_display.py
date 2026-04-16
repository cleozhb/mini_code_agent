"""计划展示与交互 — 用 Rich Table 显示 Plan，并让用户确认 / 编辑 / 放弃."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.planner import Plan, PlanStep

# 复杂度对应的颜色
_COMPLEXITY_STYLE = {
    "simple": "green",
    "medium": "yellow",
    "complex": "red",
}

PlanDecision = Literal["confirm", "cancel"]


@dataclass
class PlanConfirmation:
    """用户对计划的决定."""

    decision: PlanDecision  # confirm / cancel
    plan: Plan | None  # confirm 时返回最终的 Plan（可能被编辑过）


def render_plan(plan: Plan, console: Console) -> None:
    """用 Rich Table 渲染执行计划."""
    complexity_style = _COMPLEXITY_STYLE.get(plan.estimated_complexity, "yellow")

    header = Text.assemble(
        ("任务：", "bold"),
        (plan.goal, "bold cyan"),
        "\n",
        ("复杂度：", "bold"),
        (plan.estimated_complexity, f"bold {complexity_style}"),
        ("  ", ""),
        (f"共 {len(plan.steps)} 步", "dim"),
    )
    console.print(Panel(header, title="[bold]执行计划[/bold]", border_style="cyan"))

    table = Table(show_lines=True, expand=True)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("步骤", style="white", ratio=3)
    table.add_column("文件", style="magenta", ratio=2, overflow="fold")
    table.add_column("工具", style="yellow", ratio=1, overflow="fold")
    table.add_column("验证", style="green", ratio=2, overflow="fold")

    for i, step in enumerate(plan.steps, 1):
        table.add_row(
            str(i),
            step.description,
            "\n".join(step.files_involved) or "-",
            "\n".join(step.tools_needed) or "-",
            step.verification or "-",
        )

    console.print(table)


async def confirm_plan(
    plan: Plan,
    console: Console,
    prompt_session: PromptSession,
) -> PlanConfirmation:
    """展示计划并请求用户确认.

    Returns:
        PlanConfirmation: decision=confirm 时 plan 为最终计划（可能被编辑）
    """
    current = copy.deepcopy(plan)

    while True:
        render_plan(current, console)
        console.print(
            "\n[bold]操作：[/bold] "
            "[green]y[/green]=执行  "
            "[red]n[/red]=放弃  "
            "[yellow]e[/yellow]=编辑计划"
        )

        try:
            ans = await prompt_session.prompt_async(HTML("<b>选择 [y/n/e]: </b>"))
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]已取消[/yellow]")
            return PlanConfirmation(decision="cancel", plan=None)

        choice = (ans or "").strip().lower()
        if choice in ("y", "yes", ""):
            return PlanConfirmation(decision="confirm", plan=current)
        if choice in ("n", "no"):
            console.print("[yellow]已放弃该计划[/yellow]")
            return PlanConfirmation(decision="cancel", plan=None)
        if choice in ("e", "edit"):
            current = await _edit_plan(current, console, prompt_session)
            if not current.steps:
                console.print("[red]计划为空，已取消[/red]")
                return PlanConfirmation(decision="cancel", plan=None)
            continue

        console.print(f"[red]无法识别的选项: {choice}[/red]")


# ---------------------------------------------------------------------------
# 编辑子菜单
# ---------------------------------------------------------------------------


async def _edit_plan(
    plan: Plan,
    console: Console,
    prompt_session: PromptSession,
) -> Plan:
    """简易计划编辑菜单：删除 / 修改描述 / 追加步骤."""
    while True:
        console.print(
            "\n[bold]编辑模式：[/bold] "
            "[yellow]d <n>[/yellow]=删除第 n 步  "
            "[yellow]m <n>[/yellow]=修改第 n 步描述  "
            "[yellow]a[/yellow]=追加一步  "
            "[yellow]q[/yellow]=完成编辑"
        )
        try:
            ans = await prompt_session.prompt_async(HTML("<b>编辑 > </b>"))
        except (EOFError, KeyboardInterrupt):
            break

        cmd = (ans or "").strip()
        if not cmd:
            continue
        if cmd in ("q", "quit", "done"):
            break

        parts = cmd.split(maxsplit=1)
        op = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if op == "d":
            idx = _parse_index(arg, len(plan.steps), console)
            if idx is None:
                continue
            removed = plan.steps.pop(idx)
            console.print(f"[dim]已删除第 {idx + 1} 步：{removed.description}[/dim]")

        elif op == "m":
            idx = _parse_index(arg, len(plan.steps), console)
            if idx is None:
                continue
            try:
                new_desc = await prompt_session.prompt_async(
                    HTML("<b>新描述: </b>"),
                    default=plan.steps[idx].description,
                )
            except (EOFError, KeyboardInterrupt):
                continue
            new_desc = (new_desc or "").strip()
            if new_desc:
                plan.steps[idx].description = new_desc
                console.print(f"[dim]已更新第 {idx + 1} 步[/dim]")

        elif op == "a":
            try:
                desc = await prompt_session.prompt_async(HTML("<b>新步骤描述: </b>"))
            except (EOFError, KeyboardInterrupt):
                continue
            desc = (desc or "").strip()
            if not desc:
                console.print("[yellow]描述为空，已跳过[/yellow]")
                continue
            plan.steps.append(PlanStep(description=desc))
            console.print(f"[dim]已追加第 {len(plan.steps)} 步[/dim]")

        else:
            console.print(f"[red]无法识别的编辑命令: {op}[/red]")

    return plan


def _parse_index(arg: str, length: int, console: Console) -> int | None:
    """解析 1-based 索引参数，成功返回 0-based 下标，否则返回 None."""
    if not arg:
        console.print("[red]缺少步骤编号[/red]")
        return None
    try:
        n = int(arg)
    except ValueError:
        console.print(f"[red]步骤编号不是数字: {arg}[/red]")
        return None
    if n < 1 or n > length:
        console.print(f"[red]步骤编号越界 (1-{length})[/red]")
        return None
    return n - 1


# ---------------------------------------------------------------------------
# 进度渲染
# ---------------------------------------------------------------------------


def render_step_start(index: int, total: int, step: PlanStep, console: Console) -> None:
    """在执行每一步前渲染进度标题."""
    console.print()
    console.print(
        f"[bold cyan]▶ 步骤 {index}/{total}[/bold cyan] "
        f"[bold]{step.description}[/bold]"
    )
    if step.verification:
        console.print(f"  [dim]验证：{step.verification}[/dim]")


def render_step_done(
    index: int,
    total: int,
    step: PlanStep,
    success: bool,
    console: Console,
) -> None:
    """在执行完一步后渲染结果."""
    if success:
        console.print(f"[green]✓ 步骤 {index}/{total} 完成[/green]")
    else:
        console.print(f"[red]✗ 步骤 {index}/{total} 失败[/red]")


async def ask_replan(
    console: Console,
    prompt_session: PromptSession,
) -> Literal["replan", "continue", "abort"]:
    """某步失败时询问用户：重新规划 / 继续 / 放弃."""
    console.print(
        "[bold]该步骤失败，选择下一步：[/bold] "
        "[yellow]r[/yellow]=重新规划  "
        "[yellow]c[/yellow]=继续下一步  "
        "[yellow]a[/yellow]=放弃"
    )
    try:
        ans = await prompt_session.prompt_async(HTML("<b>选择 [r/c/a]: </b>"))
    except (EOFError, KeyboardInterrupt):
        return "abort"
    choice = (ans or "").strip().lower()
    if choice in ("r", "replan"):
        return "replan"
    if choice in ("c", "continue"):
        return "continue"
    return "abort"
