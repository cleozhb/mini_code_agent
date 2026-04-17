#!/usr/bin/env python3
"""L1-add-function 的验证脚本.

协议（见 DESIGN.md §3 validate.py）：
- cwd 是该 run 的临时 workspace（runner 在外面 chdir 好）
- 可以打印任意调试信息，runner 只解析 **最后一行 stdout** 的 JSON
- JSON 必须包含 {"passed": bool, "details": str, ...}

判定逻辑：
1. workspace/src/utils.py 能被加载
2. 其中存在一个函数，把 int 型 unix 时间戳转成 YYYY-MM-DD 格式字符串
   （用 ts=0 和 ts=1_700_000_000 两个输入交叉验证，避免常数函数骗过）
3. workspace/tests/test_utils.py 存在并用 pytest 跑通
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}")
_WORKSPACE = Path.cwd()


def _emit(payload: dict) -> None:
    # 最后一行必须是合法 JSON；先打个分隔方便眼睛扫
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _load_utils():
    utils_path = _WORKSPACE / "src" / "utils.py"
    if not utils_path.is_file():
        return None, f"找不到 src/utils.py（期望路径：{utils_path}）"

    spec = importlib.util.spec_from_file_location("utils_under_eval", utils_path)
    if spec is None or spec.loader is None:
        return None, "无法构造 utils.py 的 import spec"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"执行 utils.py 失败: {type(e).__name__}: {e}"
    return module, None


def _find_ts_function(module) -> str | None:
    """在 module 里找一个 int→YYYY-MM-DD 字符串的函数."""
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            a = fn(0)
            b = fn(1_700_000_000)
        except Exception:
            continue
        if (
            isinstance(a, str)
            and isinstance(b, str)
            and _YMD.match(a)
            and _YMD.match(b)
            and a != b
        ):
            return name
    return None


def main() -> int:
    module, err = _load_utils()
    if err:
        _emit({"passed": False, "details": err})
        return 1

    fn_name = _find_ts_function(module)
    if fn_name is None:
        _emit({
            "passed": False,
            "details": "utils.py 中找不到把 Unix 时间戳转为 YYYY-MM-DD 的函数",
        })
        return 1

    test_file = _WORKSPACE / "tests" / "test_utils.py"
    if not test_file.is_file():
        _emit({
            "passed": False,
            "details": f"缺少 tests/test_utils.py；已找到函数 {fn_name}",
            "found_function": fn_name,
        })
        return 1

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-x",
            "-q",
            "--no-header",
        ],
        cwd=str(_WORKSPACE),
        capture_output=True,
        text=True,
        timeout=30,
    )
    print("--- pytest stdout ---")
    print(proc.stdout)
    print("--- pytest stderr ---")
    print(proc.stderr)

    passed = proc.returncode == 0
    _emit({
        "passed": passed,
        "details": f"pytest exit={proc.returncode}; found_function={fn_name}",
        "found_function": fn_name,
    })
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
