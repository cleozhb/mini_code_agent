#!/usr/bin/env python3
"""L2-cache-decorator 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. `app/cache.py` 顶层必须定义 `cached` 和 `clear_cache` 两个公共符号（AST）；
2. `app/client.py` 的 `Client` 类里 `get_user` / `get_profile` / `list_orders`
   三个方法名对应的可调用，在 import 后必须是**已经被 cached 包装过的版本**
   —— 通过运行期而非 AST 判断（AST 只能粗匹配 @cached，手工 `foo = cached(foo)`
   这种写法 AST 难以可靠识别）；
3. 行为验证：同 uid 第二次调用不打 backend；不同 uid 各算一次；
   `clear_cache()` 后再调同 uid 必须重新打 backend；
4. `app/backend.py` 字节未动（它是计数桩）；
5. `tests/` 目录字节级未动；
6. `pytest tests/` 全绿。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()
_TASK_DIR = Path(__file__).resolve().parent
_ORIGINAL_WORKSPACE_DIR = _TASK_DIR / "workspace"
_ORIGINAL_TESTS_DIR = _ORIGINAL_WORKSPACE_DIR / "tests"
_ORIGINAL_BACKEND = _ORIGINAL_WORKSPACE_DIR / "app" / "backend.py"


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _module_top_level_names(tree: ast.Module) -> set[str]:
    """拿模块顶层定义/赋值/导入的所有名字."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _check_cache_module_ast() -> tuple[bool, str]:
    path = _WORKSPACE / "app" / "cache.py"
    if not path.is_file():
        return False, f"找不到 {path}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/cache.py 语法错误: {e}"
    names = _module_top_level_names(tree)
    missing = [n for n in ("cached", "clear_cache") if n not in names]
    if missing:
        return False, (
            f"app/cache.py 未定义 {missing}；当前顶层名字: {sorted(names)}"
        )
    return True, "app/cache.py 导出了 cached / clear_cache"


def _prepare_import() -> tuple[object, object, object, object]:
    """把 workspace 挂到 sys.path，清 app 缓存，import 关键模块."""
    ws = str(_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    cache_mod = importlib.import_module("app.cache")
    client_mod = importlib.import_module("app.client")
    backend_mod = importlib.import_module("app.backend")
    return cache_mod, client_mod, backend_mod, None


def _check_methods_are_wrapped(client_mod, cache_mod) -> tuple[bool, str]:
    """三个方法被 cached 包装过的粗略判据：
    - 方法对象（从类上取）不等于我们"期望它本该长什么样"是不现实的；退一步
    - 只要 Backend 计数行为满足"同 uid 第二次 0 次调"，就说明被包装了
    放到 _check_cache_behavior 里一起判；这里只做"三个方法名仍在"检查.
    """
    Client = getattr(client_mod, "Client", None)
    if Client is None:
        return False, "app/client.py 里找不到 Client 类"
    for name in ("get_user", "get_profile", "list_orders"):
        if not callable(getattr(Client, name, None)):
            return False, f"Client.{name} 不存在或不可调用"
    return True, "Client 三个方法仍存在"


def _check_cache_behavior() -> tuple[bool, str]:
    """真的 import 起来跑一遍，确认缓存命中 + 失效语义."""
    try:
        cache_mod, client_mod, backend_mod, _ = _prepare_import()
    except Exception as e:  # noqa: BLE001
        return False, f"import app 模块失败: {type(e).__name__}: {e}"

    ok, msg = _check_methods_are_wrapped(client_mod, cache_mod)
    if not ok:
        return False, msg

    clear_cache = getattr(cache_mod, "clear_cache", None)
    if not callable(clear_cache):
        return False, "app/cache.py 里的 clear_cache 不可调用"

    Backend = backend_mod.Backend
    Client = client_mod.Client

    # 清一次旧状态（可能有 import 时的污染）
    clear_cache()

    backend = Backend()
    client = Client(backend)

    # 1. 同 uid 二次调用不打 backend
    client.get_user(1)
    client.get_user(1)
    if backend.calls["get_user"] != 1:
        return False, (
            f"同 uid 二次调用应命中缓存，实际 backend.calls[get_user]="
            f"{backend.calls['get_user']}（期望 1）"
        )

    # 2. 换 uid 应实打
    client.get_user(2)
    if backend.calls["get_user"] != 2:
        return False, (
            f"不同 uid 应走 backend，实际 calls[get_user]="
            f"{backend.calls['get_user']}（期望 2）"
        )

    # 3. get_profile / list_orders 同样行为
    client.get_profile(1)
    client.get_profile(1)
    if backend.calls["get_profile"] != 1:
        return False, (
            f"get_profile 缓存未生效，calls[get_profile]="
            f"{backend.calls['get_profile']}"
        )
    client.list_orders(1)
    client.list_orders(1)
    if backend.calls["list_orders"] != 1:
        return False, (
            f"list_orders 缓存未生效，calls[list_orders]="
            f"{backend.calls['list_orders']}"
        )

    # 4. clear_cache 后重新打 backend
    clear_cache()
    client.get_user(1)
    if backend.calls["get_user"] != 3:
        return False, (
            f"clear_cache 后 get_user 未重新打 backend，calls[get_user]="
            f"{backend.calls['get_user']}（期望 3）"
        )
    client.get_profile(1)
    if backend.calls["get_profile"] != 2:
        return False, (
            f"clear_cache 后 get_profile 未重新打 backend，calls[get_profile]="
            f"{backend.calls['get_profile']}（期望 2）"
        )
    client.list_orders(1)
    if backend.calls["list_orders"] != 2:
        return False, (
            f"clear_cache 后 list_orders 未重新打 backend，calls[list_orders]="
            f"{backend.calls['list_orders']}（期望 2）"
        )

    return True, (
        f"三个方法都被 cached 正确包装；calls 终态: {backend.calls}"
    )


def _check_backend_unchanged() -> tuple[bool, str]:
    """backend.py 是计数桩，不允许动 —— 否则计数语义就变了."""
    current = _WORKSPACE / "app" / "backend.py"
    if not current.is_file():
        return False, "app/backend.py 被删除"
    if current.read_bytes() != _ORIGINAL_BACKEND.read_bytes():
        return False, "app/backend.py 被修改（内容与原始不符）"
    return True, "app/backend.py 未改"


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
            return False, f"tests/{rel} 被修改（内容与原始不符）"
    return True, "tests/ 下文件与原始一致"


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
    ok, detail = _check_cache_module_ast()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_backend_unchanged()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_cache_behavior()
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

    _emit({
        "passed": True,
        "details": "cached / clear_cache 实现正确，三个方法都生效，测试全绿",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
