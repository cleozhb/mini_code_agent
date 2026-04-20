#!/usr/bin/env python3
"""L3-find-and-fix-bugs 的验证脚本.

判定逻辑：
1. `app/` 下三个模块（discount / tax / cart）都得在，且 AST 解析成功；
2. 对外 API 签名未被偷偷改动（参数名/顺序）；
3. `tests/` 目录字节级未动；
4. `pytest tests/` 全绿（buggy 初态有 5 个失败）。

AST 阶段不检查具体"有没有加 raise ValueError"之类的实现细节 —— tests 已经
把行为钉住了；只守住"签名 + 文件结构"这一层防火墙。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()
_TASK_DIR = Path(__file__).resolve().parent
_ORIGINAL_TESTS_DIR = _TASK_DIR / "workspace" / "tests"


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _check_signatures() -> tuple[bool, str]:
    """三个 public 函数签名必须原样保留。"""
    expectations: list[tuple[str, str, list[str]]] = [
        ("app/discount.py", "apply_discount", ["price", "pct"]),
        ("app/tax.py",      "compute_tax",    ["subtotal", "rate"]),
        ("app/cart.py",     "checkout",       ["items", "discount_pct", "tax_rate"]),
    ]
    for rel, fname, expected_params in expectations:
        path = _WORKSPACE / rel
        if not path.is_file():
            return False, f"找不到 {rel}"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            return False, f"{rel} 语法错误: {e}"
        fn = _find_function(tree, fname)
        if fn is None:
            return False, f"{rel} 缺少 {fname}（被删或改名）"
        params = [a.arg for a in fn.args.args]
        if params != expected_params:
            return False, (
                f"{rel}::{fname} 签名被改：期望参数 {expected_params}，实际 {params}"
            )
    return True, "三个公共函数签名保留完好"


def _check_tests_unchanged() -> tuple[bool, str]:
    workspace_tests = _WORKSPACE / "tests"
    if not workspace_tests.is_dir():
        return False, "tests/ 目录被删除"
    for p in _ORIGINAL_TESTS_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(_ORIGINAL_TESTS_DIR)
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        current = workspace_tests / rel
        if not current.is_file():
            return False, f"tests/{rel} 被删除"
        if current.read_bytes() != p.read_bytes():
            return False, f"tests/{rel} 被修改（行为契约不允许动）"
    return True, "tests/ 下文件与原始一致"


def _run_pytest() -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(_WORKSPACE) + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
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
    ok, detail = _check_signatures()
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

    _emit({"passed": True, "details": "bug 已修，签名保留，tests 未动，全绿"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
