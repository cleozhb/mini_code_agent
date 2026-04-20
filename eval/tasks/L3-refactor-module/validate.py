#!/usr/bin/env python3
"""L3-refactor-module 的验证脚本.

判定逻辑：
1. `app/session.py` 存在并 AST 解析成功，且定义了全部 session 符号
   （Session / SessionNotFound / create_session / get_session / expire_session /
   purge_expired / reset_sessions）。
2. `app/auth.py` 的 session 相关符号已经"搬走"，顶层不再自己 def/class，
   但可以通过 `from app.session import ...` re-export。
3. `app/auth.py` 仍然在顶层 def `login` / `logout` / `verify`，且 `AuthError`
   仍在那里；login/logout/verify 的参数名不变。
4. 行为校验：login → session 存进 app.session._SESSIONS（证明 auth 真的走 session
   模块，而不是自带一份 shadow store）；verify 能拿到 user_id。
5. `tests/` 字节级未动。
6. `pytest tests/` 全绿。
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()
_TASK_DIR = Path(__file__).resolve().parent
_ORIGINAL_TESTS_DIR = _TASK_DIR / "workspace" / "tests"

_SESSION_FUNCS = {
    "create_session",
    "get_session",
    "expire_session",
    "purge_expired",
    "reset_sessions",
}
_SESSION_TYPES = {"Session", "SessionNotFound"}


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _top_level_defs(tree: ast.Module) -> tuple[set[str], set[str]]:
    """返回 (顶层函数名集合, 顶层类名集合)."""
    funcs: set[str] = set()
    classes: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
    return funcs, classes


def _imported_names(tree: ast.Module) -> set[str]:
    """收集 `from X import A, B as C` 引入的本地名。"""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _check_session_module() -> tuple[bool, str]:
    path = _WORKSPACE / "app" / "session.py"
    if not path.is_file():
        return False, "缺少 app/session.py（需要新建）"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/session.py 语法错误: {e}"
    funcs, classes = _top_level_defs(tree)
    missing_funcs = _SESSION_FUNCS - funcs
    missing_types = _SESSION_TYPES - classes
    if missing_funcs or missing_types:
        return False, (
            f"app/session.py 缺少符号：函数 {sorted(missing_funcs)}，"
            f"类 {sorted(missing_types)}"
        )
    return True, "app/session.py 符号齐全"


def _check_auth_module() -> tuple[bool, str]:
    path = _WORKSPACE / "app" / "auth.py"
    if not path.is_file():
        return False, "缺少 app/auth.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/auth.py 语法错误: {e}"
    funcs, classes = _top_level_defs(tree)

    # 1) session 函数不允许在 auth.py 顶层自己 def
    leaked_funcs = _SESSION_FUNCS & funcs
    if leaked_funcs:
        return False, (
            f"app/auth.py 顶层还自己 def 了 {sorted(leaked_funcs)}，"
            "session 的实现应该搬到 app/session.py（可以 from app.session import 再 re-export）"
        )
    # 2) Session / SessionNotFound 不允许在 auth.py 顶层用 class 重新定义
    leaked_types = _SESSION_TYPES & classes
    if leaked_types:
        return False, (
            f"app/auth.py 顶层还用 class 重复定义了 {sorted(leaked_types)}，"
            "应该从 app/session.py import 复用"
        )
    # 3) auth 仍需要保留 login / logout / verify / AuthError
    if "login" not in funcs:
        return False, "app/auth.py 顶层缺少 login"
    if "logout" not in funcs:
        return False, "app/auth.py 顶层缺少 logout"
    if "verify" not in funcs:
        return False, "app/auth.py 顶层缺少 verify"
    if "AuthError" not in classes:
        return False, "app/auth.py 顶层缺少 AuthError（定义或 re-import 都行，但不能没了）"

    # 4) 签名参数名
    sig_map: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"login", "logout", "verify"}:
            sig_map[node.name] = [a.arg for a in node.args.args]
    for fname, expected in (
        ("login",  ["user_id", "password"]),
        ("logout", ["token"]),
        ("verify", ["token"]),
    ):
        if sig_map.get(fname) != expected:
            return False, (
                f"app/auth.py::{fname} 签名被改：期望 {expected}，"
                f"实际 {sig_map.get(fname)}"
            )

    # 5) Session 名字得能从 auth 里解析出来（test_auth.py 用 `from app.auth import Session`）
    imported = _imported_names(tree)
    if "Session" not in imported and "Session" not in classes:
        return False, "app/auth.py 里 `Session` 名字不可用（tests 要 `from app.auth import Session`）"

    return True, "app/auth.py 结构正确"


def _check_runtime_wiring() -> tuple[bool, str]:
    ws = str(_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    try:
        session_mod = importlib.import_module("app.session")
        auth_mod = importlib.import_module("app.auth")
    except Exception as e:  # noqa: BLE001
        return False, f"import app.session / app.auth 失败: {type(e).__name__}: {e}"

    # auth.login 登录后，session.get_session 必须能拿到它 —— 证明只有一份 store
    if hasattr(session_mod, "reset_sessions"):
        session_mod.reset_sessions()
    try:
        s = auth_mod.login(user_id=1, password="alice_pw")
    except Exception as e:  # noqa: BLE001
        return False, f"auth.login 调用失败: {type(e).__name__}: {e}"

    try:
        same = session_mod.get_session(s.token)
    except Exception as e:  # noqa: BLE001
        return False, (
            f"session.get_session 查不到 auth.login 创建的 session："
            f"{type(e).__name__}: {e}。很可能 auth 自己又复制了一份 _SESSIONS。"
        )
    if same.user_id != 1:
        return False, f"session.get_session 返回的 user_id 不对：{same.user_id}"

    # verify 在行为上也要保留
    if auth_mod.verify(s.token) != 1:
        return False, "auth.verify 没能返回正确的 user_id"

    return True, "login/session 共享同一份存储，verify 行为保留"


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
            return False, f"tests/{rel} 被修改（不允许动 tests）"
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
    for check in (
        _check_session_module,
        _check_auth_module,
        _check_runtime_wiring,
        _check_tests_unchanged,
        _run_pytest,
    ):
        ok, detail = check()
        if not ok:
            _emit({"passed": False, "details": detail})
            return 1

    _emit({
        "passed": True,
        "details": "session 已抽出到独立模块，auth 共享同一份 _SESSIONS，测试全绿",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
