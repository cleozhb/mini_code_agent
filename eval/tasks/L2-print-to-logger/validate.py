#!/usr/bin/env python3
"""L2-print-to-logger 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. `app/service.py` 和 `app/utils.py` 的 AST 里不能再有 `print(...)` 调用；
2. 这两个文件必须有模块级 `logger` 名字（赋值或 import）；
3. `app/logging_config.py` 存在、能被 import，且 import 后 `logging.getLogger("app")`
   要么自己有 handler，要么它的祖先（root）有 handler 兜底；
4. 跑时插一个 MemoryHandler 捕获 "app" logger 的 records，然后实际调几个
   service / utils 方法，必须捕获到对应级别的 records —— 证明 print 真的换成
   logger 调用了，而不是直接删掉；
5. `tests/` 目录字节级未动（对比原始 workspace）；
6. `pytest tests/` 全绿。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from logging.handlers import MemoryHandler
from pathlib import Path

_WORKSPACE = Path.cwd()
_TASK_DIR = Path(__file__).resolve().parent
_ORIGINAL_TESTS_DIR = _TASK_DIR / "workspace" / "tests"


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _has_print_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "print":
                return True
    return False


def _has_module_level_logger(tree: ast.Module) -> bool:
    """模块顶层要有 `logger` 名字 —— 可以是赋值也可以是 from-import."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "logger":
                    return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "logger":
                return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == "logger":
                    return True
    return False


def _check_source_ast() -> tuple[bool, str]:
    for rel in ("app/service.py", "app/utils.py"):
        path = _WORKSPACE / rel
        if not path.is_file():
            return False, f"找不到 {rel}"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            return False, f"{rel} 语法错误: {e}"
        if _has_print_call(tree):
            return False, f"{rel} 里仍有 print(...) 调用"
        if not _has_module_level_logger(tree):
            return False, f"{rel} 顶部没有模块级 logger（应有 `logger = ...`）"
    return True, "service.py/utils.py 已无 print 且含模块级 logger"


def _check_logging_config_and_capture() -> tuple[bool, str]:
    """import logging_config 后用 MemoryHandler 捕获真实调用的 log records."""
    cfg_path = _WORKSPACE / "app" / "logging_config.py"
    if not cfg_path.is_file():
        return False, f"找不到 app/logging_config.py：{cfg_path}"

    # 把 workspace 放到 sys.path 最前面，`import app.*` 走工作区版本
    ws = str(_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    # 清掉残留的 app 模块缓存（validate 多次调试时）
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    try:
        importlib.import_module("app.logging_config")
    except Exception as e:
        return False, f"import app.logging_config 失败: {type(e).__name__}: {e}"

    app_logger = logging.getLogger("app")
    # 既能直接挂 handler 上也能走 root：只要有办法让 record 真的流出来就行
    has_handler = bool(app_logger.handlers) or bool(
        logging.getLogger().handlers
    )
    if not has_handler:
        return False, (
            "logging.getLogger('app') 没有任何 handler，root logger 也没有 —— "
            "logging_config 没起到配置作用"
        )

    # 插 MemoryHandler 捕获 records
    buffer = MemoryHandler(capacity=1024, flushLevel=logging.CRITICAL + 10)
    buffer.setLevel(logging.DEBUG)
    app_logger.addHandler(buffer)
    prev_level = app_logger.level
    if app_logger.level > logging.DEBUG:
        app_logger.setLevel(logging.DEBUG)
    try:
        try:
            service_mod = importlib.import_module("app.service")
            utils_mod = importlib.import_module("app.utils")
        except Exception as e:
            return False, (
                f"import app.service / app.utils 失败: {type(e).__name__}: {e}"
            )

        # 触发一系列调用；期望至少有一条 INFO（add_item 成功）、一条 WARNING
        # （add_item 非正 quantity 或 format_price 负数）、一条 ERROR（checkout 空）
        cart = service_mod.CartService()
        cart.add_item("A", 1)           # INFO
        cart.add_item("B", 0)           # WARNING
        empty = service_mod.CartService()
        empty.checkout()                 # ERROR（空购物车）
        utils_mod.format_price(-1.0)     # WARNING
        utils_mod.parse_sku("")          # ERROR
    finally:
        app_logger.removeHandler(buffer)
        app_logger.setLevel(prev_level)

    levels = {record.levelno for record in buffer.buffer}
    required = {logging.INFO, logging.WARNING, logging.ERROR}
    missing = required - levels
    if missing:
        missing_names = sorted(logging.getLevelName(lv) for lv in missing)
        return False, (
            f"MemoryHandler 没抓到这些级别的 log records: {missing_names}；"
            f"实际捕获的级别: {sorted(logging.getLevelName(lv) for lv in levels)}"
        )
    return True, (
        f"捕获到 {len(buffer.buffer)} 条 app logger records，"
        f"涵盖 INFO/WARNING/ERROR"
    )


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
    ok, detail = _check_source_ast()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_logging_config_and_capture()
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
        "details": "print 全替换为 logger、logging_config 配置生效、测试全绿",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
