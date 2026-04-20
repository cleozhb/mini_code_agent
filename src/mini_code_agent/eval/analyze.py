"""Offline trace 分析：把一个 trace.json 里的 tool 调用摊开看。

用法：
    uv run python -m mini_code_agent.eval.analyze <trace.json> [<trace.json> ...]

输出每个 trace 的：
- 顶层 metadata（task_id / stop_reason / passed / tool_calls_count / tool_calls_errors）
- tool 调用按 name 聚合：total / is_error 数量
- Bash 调用按命令头（第一个 token）聚合：total / is_error / exit code 分布
- is_error=True 的调用清单（前 N 条），content 截断到 200 字符

trace 格式约定见 runner._serialize_message：assistant msg 有 tool_calls（arguments
可能是 dict 或 JSON 字符串），tool msg 有 tool_result.{is_error,content,tool_use_id}；
tool_use_id 在旧 trace 可能为 None，所以靠"相邻 assistant → tool"配对，不是 id 匹配。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


_EXIT_RE = re.compile(r"\[exit code: (-?\d+)\]")


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _bash_head(command: str) -> str:
    """从 shell 命令里抽第一个可执行名（跳过 env 赋值和常见前缀）."""
    toks = command.strip().split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if "=" in t and not t.startswith("-"):  # FOO=bar python ...
            i += 1
            continue
        if t in ("sudo", "time", "exec"):
            i += 1
            continue
        # python -m foo → 展成 "python -m foo"
        if t in ("python", "python3") and i + 2 < len(toks) and toks[i + 1] == "-m":
            return f"{t} -m {toks[i + 2]}"
        if t == "uv" and i + 1 < len(toks) and toks[i + 1] == "run":
            j = i + 2
            if j < len(toks) and toks[j] in ("python", "python3") and j + 2 < len(toks) and toks[j + 1] == "-m":
                return f"uv run {toks[j]} -m {toks[j + 2]}"
            return "uv run"
        return t
    return "<empty>"


def _exit_from_output(content: str) -> int | None:
    m = _EXIT_RE.search(content or "")
    return int(m.group(1)) if m else None


def analyze(trace_path: Path) -> dict:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    msgs = data.get("messages", [])

    # 配对：扫描 messages，assistant.tool_calls 逐个与紧随其后的 tool 消息按顺序配对。
    pairs: list[tuple[dict, dict | None]] = []  # (tool_call, tool_result_msg)
    pending: list[dict] = []  # tool_calls 等待配对
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                pending.append(tc)
        elif role == "tool":
            tr = m.get("tool_result")
            if pending:
                pairs.append((pending.pop(0), tr))
            # 无配对 tool_call（不应发生）就丢掉
    # 剩下的 tool_call 没收到 result（比如 max_rounds 被截断）
    for tc in pending:
        pairs.append((tc, None))

    per_tool: dict[str, Counter] = defaultdict(Counter)  # name → Counter(total/err)
    bash_stats: dict[str, Counter] = defaultdict(Counter)  # head → Counter(total/err + exit_*)
    errors: list[dict] = []

    for tc, tr in pairs:
        name = tc.get("name", "<unknown>")
        is_err = bool(tr and tr.get("is_error"))
        per_tool[name]["total"] += 1
        if is_err:
            per_tool[name]["is_error"] += 1

        if name == "Bash":
            args = _parse_args(tc.get("arguments") or tc.get("input"))
            cmd = args.get("command", "")
            head = _bash_head(cmd)
            bash_stats[head]["total"] += 1
            if is_err:
                bash_stats[head]["is_error"] += 1
            exit_code = _exit_from_output((tr or {}).get("content", ""))
            if exit_code is not None:
                bash_stats[head][f"exit={exit_code}"] += 1
            elif not is_err:
                bash_stats[head]["exit=0"] += 1

        if is_err:
            errors.append({
                "tool": name,
                "args": tc.get("arguments"),
                "content": ((tr or {}).get("content") or "")[:200],
            })

    return {
        "meta": {
            k: data.get(k)
            for k in (
                "task_id", "run_index", "stop_reason", "passed",
                "tool_calls_count", "tool_calls_errors",
                "wall_time_seconds",
            )
        },
        "per_tool": {k: dict(v) for k, v in per_tool.items()},
        "bash": {k: dict(v) for k, v in bash_stats.items()},
        "errors": errors,
    }


def _print(report: dict, path: Path, max_errors: int = 10) -> None:
    meta = report["meta"]
    print(f"=== {path} ===")
    print(
        f"  task={meta['task_id']}  run={meta['run_index']}  "
        f"stop={meta['stop_reason']}  passed={meta['passed']}  "
        f"wall={meta['wall_time_seconds']:.1f}s"
    )
    print(
        f"  tool_calls_count={meta['tool_calls_count']}  "
        f"tool_calls_errors={meta['tool_calls_errors']}"
    )

    print("  per_tool:")
    for name, c in sorted(report["per_tool"].items(), key=lambda kv: -kv[1]["total"]):
        err = c.get("is_error", 0)
        print(f"    {name:12s} total={c['total']:3d}  is_error={err}")

    if report["bash"]:
        print("  bash (by command head):")
        for head, c in sorted(report["bash"].items(), key=lambda kv: -kv[1]["total"]):
            exits = sorted(
                (k, v) for k, v in c.items() if k.startswith("exit=")
            )
            exits_str = ", ".join(f"{k}:{v}" for k, v in exits) or "-"
            print(
                f"    {head:28s} total={c['total']:3d}  "
                f"is_error={c.get('is_error', 0)}  [{exits_str}]"
            )

    errs = report["errors"]
    if errs:
        print(f"  is_error 调用（前 {min(max_errors, len(errs))} / {len(errs)} 条）:")
        for e in errs[:max_errors]:
            args_str = json.dumps(e["args"], ensure_ascii=False)[:120]
            print(f"    - [{e['tool']}] args={args_str}")
            content_one = (e["content"] or "").replace("\n", " ⏎ ")
            print(f"        → {content_one}")
    print()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        p = Path(arg)
        if not p.is_file():
            print(f"not a file: {p}", file=sys.stderr)
            continue
        _print(analyze(p), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
