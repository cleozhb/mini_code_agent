#!/usr/bin/env python3
"""L4-script-to-project 验证脚本.

协议（见 DESIGN.md §3 validate.py）：
- cwd 是该 run 的临时 workspace（runner 在外面 chdir 好）
- 可以打印任意调试信息，runner 只解析 **最后一行 stdout** 的 JSON
- JSON 必须包含 {"passed": bool, "details": str, ...}

判定逻辑：
1. 检查必需的项目结构文件存在
2. core.py 包含 convert 函数
3. cli.py 使用 argparse
4. config.py 包含配置类
5. logger.py 包含 logging 模块
6. main.py 引用 converter 包
7. 存在测试文件
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()


def _emit(payload: dict) -> None:
    """最后一行必须是合法 JSON."""
    print()
    print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    errors: list[str] = []

    # 检查必需的项目结构
    required = [
        "converter/__init__.py",
        "converter/cli.py",
        "converter/config.py",
        "converter/core.py",
        "converter/logger.py",
        "converter/utils.py",
        "main.py",
    ]
    for f in required:
        if not (_WORKSPACE / f).is_file():
            errors.append(f"缺少文件: {f}")

    if errors:
        _emit({"passed": False, "details": "; ".join(errors)})
        return 1

    # 检查 core.py 包含核心转换逻辑
    core = (_WORKSPACE / "converter/core.py").read_text()
    if "convert" not in core:
        errors.append("converter/core.py 未包含 convert 函数")

    # 检查 cli.py 包含 argparse
    cli = (_WORKSPACE / "converter/cli.py").read_text()
    if "argparse" not in cli:
        errors.append("converter/cli.py 未使用 argparse")

    # 检查 config.py 包含配置类
    config = (_WORKSPACE / "converter/config.py").read_text()
    if "class" not in config.lower() and "dataclass" not in config.lower():
        errors.append("converter/config.py 未包含配置类")

    # 检查 logger.py 包含日志配置
    logger_src = (_WORKSPACE / "converter/logger.py").read_text()
    if "logging" not in logger_src:
        errors.append("converter/logger.py 未包含 logging 模块")

    # 检查 main.py 是入口
    main_src = (_WORKSPACE / "main.py").read_text()
    if "converter" not in main_src:
        errors.append("main.py 未引用 converter 包")

    # 检查测试文件
    test_dir = _WORKSPACE / "tests"
    has_tests = False
    if test_dir.is_dir():
        test_files = list(test_dir.glob("test_*.py"))
        if test_files:
            has_tests = True
    if not has_tests:
        errors.append("未找到测试文件 (tests/test_*.py)")

    if errors:
        _emit({"passed": False, "details": "; ".join(errors)})
        return 1

    _emit({"passed": True, "details": "所有检查通过"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
