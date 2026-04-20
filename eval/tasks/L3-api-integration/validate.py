#!/usr/bin/env python3
"""L3-api-integration 的验证脚本.

判定逻辑：
1. `app/http.py` 字节级未动；`tests/` 字节级未动。
2. `app/api_client.py` AST 解析成功；ApiClient 上仍然有
   `_request_with_retry` / `get_user`；`MAX_RETRIES` / `_SLEEP` 仍是模块级变量；
   `create_item` / `list_items` 方法体**不再只是 `raise NotImplementedError`**
   （用 AST 识别）。
3. `pytest tests/` 全绿（含重试次数断言）。
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
_ORIGINAL = _TASK_DIR / "workspace"
_ORIGINAL_TESTS_DIR = _ORIGINAL / "tests"
_ORIGINAL_HTTP = _ORIGINAL / "app" / "http.py"


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(
    cls: ast.ClassDef, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _is_body_just_raise_notimplemented(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """识别"函数体只剩 docstring + raise NotImplementedError"的 stub 形态."""
    body = list(fn.body)
    # 跳过开头的 docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if exc is None:
        return False
    # `raise NotImplementedError` 或 `raise NotImplementedError(...)`
    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
        return True
    if (
        isinstance(exc, ast.Call)
        and isinstance(exc.func, ast.Name)
        and exc.func.id == "NotImplementedError"
    ):
        return True
    return False


def _check_api_client_ast() -> tuple[bool, str]:
    path = _WORKSPACE / "app" / "api_client.py"
    if not path.is_file():
        return False, "缺少 app/api_client.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/api_client.py 语法错误: {e}"

    # 模块级符号不允许被删
    top_names: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    top_names.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            top_names.add(n.target.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_names.add(n.name)
    for required in ("MAX_RETRIES", "_SLEEP", "ApiError", "Item", "User", "ApiClient"):
        if required not in top_names:
            return False, f"app/api_client.py 顶层缺少 {required}（不允许删）"

    cls = _find_class(tree, "ApiClient")
    if cls is None:
        return False, "app/api_client.py 缺少 ApiClient 类"
    for mname in ("_request_with_retry", "get_user", "create_item", "list_items"):
        if _find_method(cls, mname) is None:
            return False, f"ApiClient 缺少方法 {mname}"

    for mname in ("create_item", "list_items"):
        m = _find_method(cls, mname)
        assert m is not None
        if _is_body_just_raise_notimplemented(m):
            return False, f"ApiClient.{mname} 还是 raise NotImplementedError 的 stub"

    return True, "api_client.py 结构符合预期"


def _check_bytes_unchanged(orig: Path, rel: str) -> tuple[bool, str]:
    cur = _WORKSPACE / rel
    if not cur.is_file():
        return False, f"{rel} 被删除"
    if cur.read_bytes() != orig.read_bytes():
        return False, f"{rel} 被修改（不允许动）"
    return True, ""


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
    return True, "tests/ 未改动"


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
    ok, detail = _check_bytes_unchanged(_ORIGINAL_HTTP, "app/http.py")
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_tests_unchanged()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_api_client_ast()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _run_pytest()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    _emit({
        "passed": True,
        "details": "create_item / list_items 已补完，重试 + 错误处理行为符合，测试全绿",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
