#!/usr/bin/env python3
"""L1-fix-failing-test 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. workspace/parser.py 仍然定义 parse_key_value_lines
2. workspace/tests/test_parser.py 原来的 4 个用例都还在且未被改动（按关键字签名 + 关键断言字符串）
3. 独立调用 parse_key_value_lines，在"value 含等号"的场景下行为正确
4. pytest 全部通过
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()

_REQUIRED_TESTS = (
    "test_simple_pairs",
    "test_ignores_comments_and_blank",
    "test_value_contains_equals_sign",
    "test_value_can_be_empty_string",
)

# 作弊兜底：测试文件必须原样保留这段关键断言
_REQUIRED_TEST_FRAGMENTS = (
    'url=http://example.com?a=b',
    "http://example.com?a=b",
)


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _load_parser():
    path = _WORKSPACE / "parser.py"
    if not path.is_file():
        return None, f"找不到 parser.py：{path}"
    spec = importlib.util.spec_from_file_location("parser_under_eval", path)
    if spec is None or spec.loader is None:
        return None, "无法构造 parser.py 的 import spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"执行 parser.py 失败: {type(e).__name__}: {e}"
    return module, None


def _check_tests_intact() -> tuple[bool, str]:
    test_file = _WORKSPACE / "tests" / "test_parser.py"
    if not test_file.is_file():
        return False, f"找不到 tests/test_parser.py：{test_file}"
    content = test_file.read_text(encoding="utf-8")
    for name in _REQUIRED_TESTS:
        if not re.search(rf"\bdef\s+{re.escape(name)}\s*\(", content):
            return False, f"测试用例 {name} 被删除或改名（不允许修改测试）"
    for fragment in _REQUIRED_TEST_FRAGMENTS:
        if fragment not in content:
            return False, f"测试文件中关键断言被修改：缺少片段 {fragment!r}"
    return True, "测试文件完整"


def main() -> int:
    module, err = _load_parser()
    if err:
        _emit({"passed": False, "details": err})
        return 1

    fn = getattr(module, "parse_key_value_lines", None)
    if fn is None or not callable(fn):
        _emit({"passed": False, "details": "parser.py 中找不到 parse_key_value_lines"})
        return 1

    # 独立验证行为（防止只是改 test 使其 pass）
    try:
        result = fn("url=http://example.com?a=b")
    except Exception as e:
        _emit({
            "passed": False,
            "details": f"parse_key_value_lines 处理含等号 value 时抛异常: {type(e).__name__}: {e}",
        })
        return 1
    if result != {"url": "http://example.com?a=b"}:
        _emit({
            "passed": False,
            "details": f"含等号 value 场景结果错误：{result!r}",
        })
        return 1

    ok, detail = _check_tests_intact()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_WORKSPACE / "tests" / "test_parser.py"),
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
        "details": f"pytest exit={proc.returncode}; parser + tests intact",
    })
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
