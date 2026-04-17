#!/usr/bin/env python3
"""L1-add-docstring 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. workspace/math_helpers.py 存在、能被 ast 解析
2. 原有的 5 个公共函数（add/subtract/multiply/divide/clamp）都还在
3. 每个公共函数的 ast.get_docstring(...) 非空、长度 >= 10（不允许水一行）
4. 函数签名未改（参数名顺序不变）
5. 行为未变：add(1,2)==3；divide(1,0) 抛 ValueError；clamp(5,0,10)==5；clamp(-1,0,10)==0；clamp(99,0,10)==10
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()

_EXPECTED_FUNCTIONS = {
    "add": ["a", "b"],
    "subtract": ["a", "b"],
    "multiply": ["a", "b"],
    "divide": ["a", "b"],
    "clamp": ["value", "low", "high"],
}

_MIN_DOCSTRING_LEN = 10


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _check_docstrings_in_source() -> tuple[bool, str]:
    path = _WORKSPACE / "math_helpers.py"
    if not path.is_file():
        return False, f"找不到 math_helpers.py：{path}"
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"math_helpers.py 语法错误: {e}"

    funcs_by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name, expected_params in _EXPECTED_FUNCTIONS.items():
        node = funcs_by_name.get(name)
        if node is None:
            return False, f"函数 {name} 被删除"
        actual_params = [arg.arg for arg in node.args.args]
        if actual_params != expected_params:
            return False, (
                f"函数 {name} 签名被改动：期望 {expected_params}，实际 {actual_params}"
            )
        doc = ast.get_docstring(node)
        if not doc or not doc.strip():
            return False, f"函数 {name} 没有 docstring"
        if len(doc.strip()) < _MIN_DOCSTRING_LEN:
            return False, (
                f"函数 {name} 的 docstring 过短（{len(doc.strip())} 字符，"
                f"要求 >= {_MIN_DOCSTRING_LEN}）"
            )
    return True, "5 个函数都已有 docstring 且签名未改"


def _check_behavior_unchanged() -> tuple[bool, str]:
    path = _WORKSPACE / "math_helpers.py"
    spec = importlib.util.spec_from_file_location("math_helpers_under_eval", path)
    if spec is None or spec.loader is None:
        return False, "无法构造 math_helpers.py 的 import spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"执行 math_helpers.py 失败: {type(e).__name__}: {e}"

    if module.add(1, 2) != 3:
        return False, f"add(1,2) 期望 3，实际 {module.add(1, 2)!r}"
    if module.subtract(5, 3) != 2:
        return False, f"subtract(5,3) 期望 2，实际 {module.subtract(5, 3)!r}"
    if module.multiply(4, 3) != 12:
        return False, f"multiply(4,3) 期望 12，实际 {module.multiply(4, 3)!r}"
    if module.divide(10, 2) != 5:
        return False, f"divide(10,2) 期望 5，实际 {module.divide(10, 2)!r}"

    try:
        module.divide(1, 0)
    except ValueError:
        pass
    except Exception as e:
        return False, f"divide(1,0) 抛了非 ValueError 异常：{type(e).__name__}"
    else:
        return False, "divide(1,0) 应该抛 ValueError"

    if module.clamp(5, 0, 10) != 5:
        return False, f"clamp(5,0,10) 期望 5，实际 {module.clamp(5, 0, 10)!r}"
    if module.clamp(-1, 0, 10) != 0:
        return False, f"clamp(-1,0,10) 期望 0，实际 {module.clamp(-1, 0, 10)!r}"
    if module.clamp(99, 0, 10) != 10:
        return False, f"clamp(99,0,10) 期望 10，实际 {module.clamp(99, 0, 10)!r}"
    return True, "所有公共函数行为未变"


def main() -> int:
    ok, detail = _check_docstrings_in_source()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    ok, detail = _check_behavior_unchanged()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    _emit({"passed": True, "details": "5 函数全部有合格 docstring，签名与行为未改"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
