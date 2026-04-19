#!/usr/bin/env python3
"""L2-add-model-field 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. `app/models.py` 的 `User` 类必须有注解 `email`；
2. `app/schemas.py` 的 `UserCreateIn` / `UserOut` 都必须有注解 `email`；
3. `email` 是必填（没有默认值 / 默认工厂） —— 通过 AST 检查；
4. 真的 import 起来跑一次 create_user + get_user，email 必须能往返；
5. `tests/` 目录字节级未动；
6. `pytest tests/` 全绿。

单独做 AST "必填" 检查的理由：光靠 pytest 判不出来 —— 如果 Agent 给 email 加了
默认值 ""，pytest 的 `test_email_is_required_on_create_in` 会挂，但若 Agent 干脆
去改了 tests 就能绕过（虽然我们有 tests 字节守卫，但双保险更稳）。
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
_ORIGINAL_TESTS_DIR = _TASK_DIR / "workspace" / "tests"


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_annotated_fields(cls: ast.ClassDef) -> dict[str, ast.AnnAssign]:
    """返回 class 体内 `name: T [= default]` 形式的字段."""
    out: dict[str, ast.AnnAssign] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def _check_class_has_required_email(
    file_label: str,
    cls_name: str,
    cls: ast.ClassDef | None,
) -> tuple[bool, str]:
    if cls is None:
        return False, f"{file_label} 里找不到类 {cls_name}"
    fields = _class_annotated_fields(cls)
    if "email" not in fields:
        return False, (
            f"{file_label}.{cls_name} 没有 email 注解字段；当前字段: "
            f"{sorted(fields.keys())}"
        )
    field = fields["email"]
    if field.value is not None:
        # 有默认值 → 非必填
        try:
            default_src = ast.unparse(field.value)
        except Exception:  # noqa: BLE001
            default_src = "<unknown>"
        return False, (
            f"{file_label}.{cls_name}.email 不应该有默认值（必填字段），"
            f"当前默认: {default_src}"
        )
    return True, ""


def _check_source_ast() -> tuple[bool, str]:
    # models.py / User
    models_path = _WORKSPACE / "app" / "models.py"
    if not models_path.is_file():
        return False, f"找不到 {models_path}"
    try:
        models_tree = ast.parse(models_path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/models.py 语法错误: {e}"

    ok, msg = _check_class_has_required_email(
        "app/models.py", "User", _find_class(models_tree, "User")
    )
    if not ok:
        return False, msg

    # schemas.py / UserCreateIn, UserOut
    schemas_path = _WORKSPACE / "app" / "schemas.py"
    if not schemas_path.is_file():
        return False, f"找不到 {schemas_path}"
    try:
        schemas_tree = ast.parse(schemas_path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/schemas.py 语法错误: {e}"

    for cls_name in ("UserCreateIn", "UserOut"):
        ok, msg = _check_class_has_required_email(
            "app/schemas.py", cls_name, _find_class(schemas_tree, cls_name)
        )
        if not ok:
            return False, msg

    # api.py 必须还在，且 create_user / get_user 依旧存在、签名未变
    api_path = _WORKSPACE / "app" / "api.py"
    if not api_path.is_file():
        return False, f"找不到 {api_path}"
    try:
        api_tree = ast.parse(api_path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"app/api.py 语法错误: {e}"

    funcs = {
        n.name: n
        for n in api_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for fname, expected_params in (
        ("create_user", ["data"]),
        ("get_user", ["uid"]),
    ):
        if fname not in funcs:
            return False, f"app/api.py 缺少 {fname}（被删或改名）"
        params = [a.arg for a in funcs[fname].args.args]
        if params != expected_params:
            return False, (
                f"app/api.py::{fname} 签名被改：期望参数 {expected_params}，"
                f"实际 {params}"
            )

    return True, "三个类都有 email 必填字段；api 签名保留"


def _check_roundtrip_behavior() -> tuple[bool, str]:
    """真的 import + 调用一次，确认 email 真的流通 model/schema/api 三层."""
    ws = str(_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    try:
        api = importlib.import_module("app.api")
        schemas = importlib.import_module("app.schemas")
    except Exception as e:  # noqa: BLE001
        return False, f"import app.api/app.schemas 失败: {type(e).__name__}: {e}"

    # 清一下 store（validate 独立于测试 fixture）
    if hasattr(api, "reset_store"):
        api.reset_store()

    try:
        payload = schemas.UserCreateIn(id=777, name="Val", email="val@x.com")
    except TypeError as e:
        return False, f"UserCreateIn 不接受 email 关键字: {e}"

    try:
        out = api.create_user(payload)
    except Exception as e:  # noqa: BLE001
        return False, f"create_user 抛异常: {type(e).__name__}: {e}"

    if not hasattr(out, "email"):
        return False, "create_user 的返回值没有 email 属性"
    if out.email != "val@x.com":
        return False, (
            f"create_user 返回的 email 不对：期望 'val@x.com'，实际 {out.email!r}"
        )

    try:
        got = api.get_user(777)
    except Exception as e:  # noqa: BLE001
        return False, f"get_user 抛异常: {type(e).__name__}: {e}"
    if getattr(got, "email", None) != "val@x.com":
        return False, (
            f"get_user 返回的 email 不对：期望 'val@x.com'，"
            f"实际 {getattr(got, 'email', None)!r}"
        )

    return True, "email 通过 create_user / get_user 往返成功"


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

    ok, detail = _check_roundtrip_behavior()
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
        "details": "email 贯通 model/schema/api 三层，测试全绿",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
