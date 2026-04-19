#!/usr/bin/env python3
"""L2-split-large-function 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. workspace/order_processor.py 能被 ast 解析；
2. 所有模块级 FunctionDef 的"函数体行数"必须 ≤ _MAX_BODY_LINES（默认 30）；
3. 必须保留 `process_order` 函数，且签名未改（参数名 + 顺序）；
4. 必须真的拆过：模块级函数数量 ≥ _MIN_FUNCTION_COUNT（否则就是没拆）；
5. 四个业务常量 TIER_DISCOUNTS / TAX_RATES / COUPON_DISCOUNTS / SHIPPING_BASE
   的数值必须与原始一致（Agent 不应为了过测试改常量）；
6. tests/ 目录没被改动（sha256 对比原始 workspace），Agent 只能改实现；
7. pytest 跑 tests/ 全绿。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()
# validate.py 跑在 workspace 的临时副本里；__file__ 仍指向 task 目录下的
# 本文件，借此拿到原始 workspace/tests 做 byte-for-byte 对比。
_TASK_DIR = Path(__file__).resolve().parent
_ORIGINAL_TESTS_DIR = _TASK_DIR / "workspace" / "tests"

_MAX_BODY_LINES = 30
_MIN_FUNCTION_COUNT = 3

_EXPECTED_PROCESS_ORDER_PARAMS = ["order"]

_EXPECTED_CONSTANTS = {
    "TIER_DISCOUNTS": {"standard": 0.0, "gold": 0.05, "vip": 0.12},
    "TAX_RATES": {"CN": 0.13, "US": 0.07, "EU": 0.20},
    "COUPON_DISCOUNTS": {"SAVE10": 0.10, "SAVE20": 0.20, "FREESHIP": 0.0},
    "SHIPPING_BASE": {"CN": 10.0, "US": 20.0, "EU": 25.0},
}

def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _function_body_line_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """函数体行数：从签名下一行到函数末尾，跳过首条 docstring。

    我们希望评判的是"业务代码行数"而不是"整个节点物理行数"。具体：
    - 忽略函数头本身（不算 def 那一行）
    - 如果首条 statement 是 docstring，跳过它
    - 其它 statement 的 end_lineno - lineno + 1 用 first/last 覆盖区间计
    """
    body = list(node.body)
    if not body:
        return 0
    # 跳过首条 docstring
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
        if not body:
            return 0
    first = body[0].lineno
    last = max(getattr(s, "end_lineno", s.lineno) for s in body)
    return last - first + 1


def _check_source() -> tuple[bool, str]:
    path = _WORKSPACE / "order_processor.py"
    if not path.is_file():
        return False, f"找不到 order_processor.py：{path}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"order_processor.py 语法错误: {e}"

    functions = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) < _MIN_FUNCTION_COUNT:
        return False, (
            f"只有 {len(functions)} 个顶层函数，未达拆分下限 {_MIN_FUNCTION_COUNT}；"
            "显然还没拆开"
        )

    oversized = [
        (f.name, _function_body_line_count(f))
        for f in functions
        if _function_body_line_count(f) > _MAX_BODY_LINES
    ]
    if oversized:
        desc = "; ".join(f"{n}({c}行)" for n, c in oversized)
        return False, (
            f"以下函数体超过 {_MAX_BODY_LINES} 行: {desc}"
        )

    # 必须有 process_order 且签名未改
    process_order = next((f for f in functions if f.name == "process_order"), None)
    if process_order is None:
        return False, "process_order 被删除或重命名"
    params = [a.arg for a in process_order.args.args]
    if params != _EXPECTED_PROCESS_ORDER_PARAMS:
        return False, (
            f"process_order 签名被改：期望参数 {_EXPECTED_PROCESS_ORDER_PARAMS}，"
            f"实际 {params}"
        )

    # 常量未改
    consts: dict[str, dict] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in _EXPECTED_CONSTANTS:
            try:
                consts[name] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return False, f"常量 {name} 的值无法求值（应为字面量 dict）"
    for name, expected in _EXPECTED_CONSTANTS.items():
        if name not in consts:
            return False, f"常量 {name} 丢失"
        if consts[name] != expected:
            return False, (
                f"常量 {name} 被改动：期望 {expected}，实际 {consts[name]}"
            )

    return True, (
        f"{len(functions)} 个顶层函数，全部 ≤ {_MAX_BODY_LINES} 行；"
        "process_order 签名保留；常量未改"
    )


def _check_tests_unchanged() -> tuple[bool, str]:
    """对比 workspace/tests/ 与原始 task_dir/workspace/tests/ 按字节是否一致."""
    workspace_tests = _WORKSPACE / "tests"
    if not workspace_tests.is_dir():
        return False, "tests 目录被删除"

    # 收集原始 tests/ 下所有文件（排除 __pycache__/.pyc）
    original_files: list[Path] = []
    for p in _ORIGINAL_TESTS_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(_ORIGINAL_TESTS_DIR)
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        original_files.append(p)

    for original in original_files:
        rel = original.relative_to(_ORIGINAL_TESTS_DIR)
        current = workspace_tests / rel
        if not current.is_file():
            return False, f"tests/{rel} 被删除"
        if current.read_bytes() != original.read_bytes():
            return False, f"tests/{rel} 被修改（内容与原始不符）"
    return True, f"tests/ 下 {len(original_files)} 个文件与原始一致"


def _run_pytest() -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(_WORKSPACE) + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/",
            "-x", "-q", "--no-header",
        ],
        cwd=str(_WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    print("--- pytest stdout ---")
    print(proc.stdout)
    print("--- pytest stderr ---")
    print(proc.stderr)
    if proc.returncode != 0:
        return False, f"pytest 失败（exit={proc.returncode}）"
    return True, "pytest 全绿"


def main() -> int:
    ok, detail = _check_source()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_tests_unchanged()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _run_pytest()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    _emit({"passed": True, "details": "重构后函数行数达标、行为未变、测试全绿"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
