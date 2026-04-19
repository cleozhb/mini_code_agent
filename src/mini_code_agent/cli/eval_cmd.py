"""`main.py eval` 子命令实现.

按 DESIGN.md §6 的 CLI 契约：
- `eval` 跑 benchmark 任务（默认 n=3 次/任务）
- `eval --level / --task / --tag` 过滤
- `eval --runs / --parallel` 调节
- `eval --no-save` 不落盘
- `eval --compare [A B]` 对比历史
- `eval --trend [N]` 最近 N 次趋势

CLI 负责的唯一 agent-factory 组装（BashTool cwd、FileGuard work_dir、
LoopGuard 按 task 限额、Agent max_wall_time_seconds）也在这里。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from ..core import Agent
from ..eval import (
    BenchmarkSuite,
    BenchmarkTask,
    EvalRunner,
    EvalTracker,
    KNOWN_MODELS,
    ModelPricing,
    SuiteResult,
    compute_suite_hash,
    pricing_for,
)
from ..llm import LLMClient, create_client
from ..safety import FileGuard, LoopGuard
from ..tools import (
    BashTool,
    EditFileTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    ToolRegistry,
    WriteFileTool,
)


# 评估时给 Agent 用的 system prompt —— 不泄漏任务思路，只强调"用工具改文件，简短总结"
_EVAL_PROMPT_FILE_LIST_LIMIT = 30  # 超过就截断，避免 prompt 过大
_EVAL_PROMPT_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def _list_workspace_files(workspace: Path) -> list[str]:
    """列出 workspace 里的相对路径，过滤掉 pycache 等噪音目录."""
    out: list[str] = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        if any(part in _EVAL_PROMPT_IGNORED_DIRS for part in p.relative_to(workspace).parts):
            continue
        out.append(str(p.relative_to(workspace)))
    return out


def _build_eval_system_prompt(workspace: Path) -> str:
    """动态生成 eval 用 system prompt：告诉模型 cwd 在哪、初始文件有哪些.

    之前用常量 system prompt 的版本在弱模型（如 DeepSeek）上迷路率很高：
    L1 任务 11 轮一个 WriteFile 都没调用成功、prompt token 爆掉。
    根因是模型不知道 cwd、不知道工作区文件结构，只能靠瞎摸。
    REPL 模式靠 `build_system_prompt_with_context` 补上这层，eval 之前绕过了。
    """
    files = _list_workspace_files(workspace)
    if not files:
        file_list = "  （工作区为空，可自由创建文件）"
    else:
        shown = files[: _EVAL_PROMPT_FILE_LIST_LIMIT]
        lines = [f"  - {f}" for f in shown]
        if len(files) > _EVAL_PROMPT_FILE_LIST_LIMIT:
            lines.append(f"  - ...（另有 {len(files) - _EVAL_PROMPT_FILE_LIST_LIMIT} 个文件未列出，可用 ListDir/Grep 自行查看）")
        file_list = "\n".join(lines)

    return (
        "你是编程 Agent，正在执行一项自动化评估任务。\n"
        "\n"
        f"当前工作目录（cwd）: {workspace}\n"
        "工作区初始文件：\n"
        f"{file_list}\n"
        "\n"
        "工具使用要点：\n"
        "- WriteFile / EditFile 的 path 参数用**相对路径**（如 'config.py'、'tests/test_config.py'）\n"
        "- Bash 执行时 cwd 已经锁在工作区里，不要 cd 出去\n"
        "- workspace 是隔离的临时副本，放开手改文件即可\n"
        "\n"
        "提交前自测（重要）：\n"
        "- 改完代码后，**必须**用 Bash 跑一次测试确认没回归，再给最终回复：\n"
        "  - 如果工作区有 `tests/` 目录：`python -m pytest tests/ -q`\n"
        "  - 如果是单文件任务（根目录下有 `test_*.py`）：`python -m pytest -q`\n"
        "- 测试挂了就继续修，通过后再收尾。不要跳过这一步直接汇报。\n"
        "- 汇报时一句话说明做了什么 + 测试结果（如 `改了 X，pytest 全绿`），不要冗长解释。\n"
    )


# ---------------------------------------------------------------------------
# argparse 配置
# ---------------------------------------------------------------------------


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在顶层 `subparsers` 上挂 `eval` 子命令."""
    p = subparsers.add_parser(
        "eval",
        help="跑 benchmark 评估，或查历史（--compare / --trend）",
        description="mini-code-agent 自带的 benchmark 评估命令",
    )
    # 过滤
    p.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                   help="只跑指定 level")
    p.add_argument("--task", type=str, default=None,
                   help="只跑指定 task id")
    p.add_argument("--tag", type=str, default=None,
                   help="只跑带指定 tag 的任务")
    # 跑多少次
    p.add_argument("--runs", type=int, default=3,
                   help="每个任务跑几次（默认 3；信噪比默认值，详见 DESIGN §9.3）")
    p.add_argument("--parallel", type=int, default=1,
                   help="任务间并发度（默认 1，串行）")
    p.add_argument("--no-save", action="store_true",
                   help="不把结果落盘")
    p.add_argument("--no-trace", action="store_true",
                   help="不落 Agent 对话 trace（默认落到 results_dir/traces/）")
    # 目录
    p.add_argument("--tasks-dir", type=str, default="eval/tasks",
                   help="benchmark 任务目录（相对 cwd，默认 eval/tasks）")
    p.add_argument("--results-dir", type=str, default="eval/results",
                   help="结果落盘目录（默认 eval/results）")
    # provider / model
    p.add_argument("--provider", type=str, default="openai",
                   choices=["openai", "anthropic"],
                   help="LLM 服务商（默认 openai）")
    p.add_argument("--model", type=str, default=None,
                   help="模型名；不指定则读 .env")
    # 分析模式
    p.add_argument("--compare", nargs="*", default=None, metavar="RUN",
                   help="对比模式：不带参数 → 最近两次；带 2 个参数 → 指定两次")
    p.add_argument("--trend", type=int, nargs="?", const=10, default=None,
                   metavar="N",
                   help="趋势模式：显示最近 N 次（默认 10）的 sparkline")
    return p


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def run_eval_command(args: argparse.Namespace) -> int:
    """`main.py eval` 的实际入口，返回进程 exit code.

    退出码约定：
    - 0 : 成功（compare/trend 正常渲染；或 eval 全部任务 100% 通过）
    - 1 : eval 跑了但有失败
    - 2 : 用户错误（参数、找不到 run、找不到 task 等）
    """
    console = Console()
    results_dir = Path(args.results_dir)
    tasks_dir = Path(args.tasks_dir)

    # 分析模式优先：compare/trend 不跑任何 agent
    if args.compare is not None:
        return _run_compare(args, results_dir, console)
    if args.trend is not None:
        return _run_trend(args, results_dir, console)

    return await _run_eval(args, tasks_dir, results_dir, console)


# ---------------------------------------------------------------------------
# compare / trend
# ---------------------------------------------------------------------------


def _run_compare(
    args: argparse.Namespace,
    results_dir: Path,
    console: Console,
) -> int:
    tracker = EvalTracker(results_dir)
    runs = tracker.list_runs()
    if len(runs) < 2:
        console.print(
            f"[red]至少需要 2 次 eval 记录才能对比，当前 {len(runs)} 次[/red]"
        )
        return 2

    if not args.compare:
        a, b = runs[-2], runs[-1]
    elif len(args.compare) == 2:
        a = _find_run(runs, args.compare[0], console)
        b = _find_run(runs, args.compare[1], console)
        if a is None or b is None:
            return 2
    else:
        console.print(
            "[red]--compare 要么不带参数（对比最近两次），要么给 2 个 run 标识[/red]"
        )
        return 2

    rep = tracker.compare(a, b, label_a=a.timestamp, label_b=b.timestamp)
    console.print(rep.render())
    return 0


def _run_trend(
    args: argparse.Namespace,
    results_dir: Path,
    console: Console,
) -> int:
    tracker = EvalTracker(results_dir)
    rep = tracker.trend(last_n=args.trend)
    console.print(rep.render())
    return 0


def _find_run(
    runs: list[SuiteResult],
    needle: str,
    console: Console,
) -> SuiteResult | None:
    """按 timestamp 子串或 commit 前缀模糊匹配 run."""
    matches = [
        r for r in runs
        if needle in r.timestamp or (r.git_commit and needle in r.git_commit)
    ]
    if not matches:
        console.print(f"[red]找不到匹配的 run: {needle}[/red]")
        return None
    if len(matches) > 1:
        console.print(
            f"[red]{len(matches)} 个 run 都匹配 {needle!r}，请给更精确的标识[/red]"
        )
        return None
    return matches[0]


# ---------------------------------------------------------------------------
# eval 主流程
# ---------------------------------------------------------------------------


async def _run_eval(
    args: argparse.Namespace,
    tasks_dir: Path,
    results_dir: Path,
    console: Console,
) -> int:
    # 1. 载入 suite + 过滤
    if not tasks_dir.is_dir():
        console.print(f"[red]tasks 目录不存在: {tasks_dir}[/red]")
        return 2
    suite = BenchmarkSuite.load_from_dir(tasks_dir)
    if args.level is not None:
        suite = suite.filter_by_level(args.level)
    if args.tag is not None:
        suite = suite.filter_by_tag(args.tag)
    if args.task is not None:
        t = suite.get(args.task)
        if t is None:
            console.print(f"[red]任务不存在: {args.task}[/red]")
            return 2
        suite = BenchmarkSuite(tasks=[t], suite_hash=compute_suite_hash([t]))
    if not suite.tasks:
        console.print("[yellow]过滤后没有任务可跑[/yellow]")
        return 2

    # 2. LLM client + 定价
    try:
        llm_client = create_client(provider=args.provider, model=args.model)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]创建 LLM 客户端失败: {e}[/red]")
        return 2
    model_name = args.model or llm_client.model
    pricing = _resolve_pricing(model_name, console)

    # 3. factory + runner
    factory = build_agent_factory(llm_client)
    # --no-save 隐含 --no-trace：既然连 summary 都不落盘，就别留调试副产物
    trace_dir = (
        None if (args.no_trace or args.no_save)
        else (results_dir / "traces")
    )
    runner = EvalRunner(
        agent_factory=factory,
        model_name=model_name,
        pricing=pricing,
        runs_per_task=args.runs,
        parallel_tasks=args.parallel,
        trace_dir=trace_dir,
    )
    total_runs = len(suite.tasks) * max(1, args.runs)
    console.print(
        f"[dim]跑 {len(suite.tasks)} 个任务 × {args.runs} run = "
        f"{total_runs} 次 · model={model_name} · parallel={args.parallel}[/dim]"
    )

    # 4. 跑 + 渲染 + 落盘
    result = await runner.run_suite(suite)
    _render_suite_result(result, console)

    if not args.no_save:
        tracker = EvalTracker(results_dir)
        path = tracker.save(result)
        console.print(f"[dim]已保存: {path}[/dim]")
        if trace_dir is not None:
            console.print(f"[dim]trace 目录: {trace_dir}[/dim]")
    else:
        console.print("[dim]--no-save：结果未落盘[/dim]")

    # 全部任务 100% 通过才算 0；否则 1，方便 CI 失败
    return 0 if result.summary.task_success_rate >= 1.0 else 1


def _resolve_pricing(model_name: str, console: Console) -> ModelPricing:
    try:
        return pricing_for(model_name)
    except KeyError:
        known = ", ".join(sorted(KNOWN_MODELS))
        console.print(
            f"[yellow]⚠ 未知模型 {model_name!r}（已知: {known}）；"
            f"cost_usd 将按 0 计[/yellow]"
        )
        return ModelPricing(0.0, 0.0)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent_factory(
    llm_client: LLMClient,
    *,
    system_prompt_override: str | None = None,
) -> Callable[[Path, BenchmarkTask], Agent]:
    """按 DESIGN §4 的要求组装 factory：把 task 的三项上限绑到 Agent 的安全层.

    `system_prompt_override` 不传时，会按每个 workspace 动态生成 system prompt
    （见 `_build_eval_system_prompt`）。传字符串则固定用它（主要给测试用）。
    """

    def factory(workspace: Path, task: BenchmarkTask) -> Agent:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(BashTool(cwd=str(workspace)))
        registry.register(GrepTool())
        registry.register(ListDirTool())

        file_guard = FileGuard(work_dir=workspace)
        loop_guard = LoopGuard(
            max_rounds=task.max_steps,
            max_tokens=task.max_tokens,
        )
        prompt = (
            system_prompt_override
            if system_prompt_override is not None
            else _build_eval_system_prompt(workspace)
        )
        return Agent(
            llm_client=llm_client,
            tool_registry=registry,
            system_prompt=prompt,
            file_guard=file_guard,
            loop_guard=loop_guard,
            max_wall_time_seconds=task.max_wall_time_seconds,
        )

    return factory


# ---------------------------------------------------------------------------
# 结果渲染
# ---------------------------------------------------------------------------


def _render_suite_result(result: SuiteResult, console: Console) -> None:
    """把一次 SuiteResult 的关键信息渲染成两张表 + 一段 summary."""
    s = result.summary

    # 1. Summary 指标
    tbl = Table(title=f"Eval Summary · {result.model_name} · {result.timestamp}")
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    rows: list[tuple[str, str]] = [
        ("task_success_rate",        f"{s.task_success_rate:.3f}"),
        ("tool_error_rate",          f"{s.tool_error_rate:.3f}"),
        ("verifier_first_pass_rate", f"{s.verifier_first_pass_rate:.3f}"),
        ("verifier_recovery_rate",   f"{s.verifier_recovery_rate:.3f}"),
        ("avg_step_count",           f"{s.avg_step_count:.2f}"),
        ("avg_prompt_tokens",        f"{s.avg_prompt_tokens:.0f}"),
        ("avg_completion_tokens",    f"{s.avg_completion_tokens:.0f}"),
        ("total_cost_usd",           f"${s.total_cost_usd:.4f}"),
        ("avg_wall_time_seconds",    f"{s.avg_wall_time_seconds:.2f}"),
        ("avg_edit_precision",       f"{s.avg_edit_precision:.3f}"),
        ("avg_edit_recall",          f"{s.avg_edit_recall:.3f}"),
    ]
    for k, v in rows:
        tbl.add_row(k, v)
    console.print(tbl)

    if s.by_level:
        lv = Table(title="Pass rate by level")
        lv.add_column("level")
        lv.add_column("rate", justify="right")
        for level in sorted(s.by_level):
            lv.add_row(f"L{level}", f"{s.by_level[level]:.3f}")
        console.print(lv)

    if s.by_failure_category:
        fc = Table(title="Failures by category")
        fc.add_column("category")
        fc.add_column("count", justify="right")
        for cat in sorted(s.by_failure_category):
            fc.add_row(cat, str(s.by_failure_category[cat]))
        console.print(fc)

    # 2. Per-run 细节（每个 task_id 的 n 次 run 汇总一行）
    per_task: dict[str, list] = {}
    for r in result.results:
        per_task.setdefault(r.task_id, []).append(r)

    pt = Table(title="Per-task")
    pt.add_column("task")
    pt.add_column("pass", justify="right")
    pt.add_column("runs", justify="right")
    pt.add_column("avg_steps", justify="right")
    pt.add_column("avg_cost", justify="right")
    pt.add_column("avg_wall", justify="right")
    pt.add_column("failures", justify="left")
    for tid in sorted(per_task):
        rs = per_task[tid]
        pass_n = sum(1 for r in rs if r.passed)
        fails = sorted({r.failure_category for r in rs if r.failure_category})
        pt.add_row(
            tid,
            f"{pass_n}/{len(rs)}",
            str(len(rs)),
            f"{sum(r.step_count for r in rs) / len(rs):.1f}",
            f"${sum(r.cost_usd for r in rs) / len(rs):.4f}",
            f"{sum(r.wall_time_seconds for r in rs) / len(rs):.1f}s",
            ", ".join(fails) if fails else "-",
        )
    console.print(pt)
