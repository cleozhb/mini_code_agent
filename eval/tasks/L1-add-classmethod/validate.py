#!/usr/bin/env python3
"""L1-add-classmethod 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. workspace/config.py 能被加载，其中有 Config 类
2. Config 类有 from_env() 类方法
3. 在受控环境变量下调用 Config.from_env()：
   - 无环境变量 → host='localhost', port=8080, debug=False
   - APP_HOST='h', APP_PORT='9001', APP_DEBUG='true' → 正确转型
   - APP_DEBUG='0' → debug=False（不是所有非空字符串都 True）
4. workspace/test_config.py 存在并 pytest 跑通
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _load_config():
    cfg_path = _WORKSPACE / "config.py"
    if not cfg_path.is_file():
        return None, f"找不到 config.py：{cfg_path}"
    spec = importlib.util.spec_from_file_location("config_under_eval", cfg_path)
    if spec is None or spec.loader is None:
        return None, "无法构造 config.py 的 import spec"
    module = importlib.util.module_from_spec(spec)
    # 先注册到 sys.modules —— @dataclass 需要能通过 __module__ 反查到模块
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"执行 config.py 失败: {type(e).__name__}: {e}"
    return module, None


def _check_from_env(Config) -> tuple[bool, str]:
    if not inspect.isclass(Config):
        return False, "Config 不是类"
    if not hasattr(Config, "from_env"):
        return False, "Config 没有 from_env 方法"

    # 清空相关 env var 再测默认
    for k in ("APP_HOST", "APP_PORT", "APP_DEBUG"):
        os.environ.pop(k, None)
    try:
        default = Config.from_env()
    except Exception as e:
        return False, f"Config.from_env() 默认场景抛异常: {type(e).__name__}: {e}"

    checks = [
        (getattr(default, "host", None) == "localhost",
         f"默认 host 应为 'localhost'，实际 {getattr(default, 'host', None)!r}"),
        (getattr(default, "port", None) == 8080,
         f"默认 port 应为 8080，实际 {getattr(default, 'port', None)!r}"),
        (getattr(default, "debug", None) is False,
         f"默认 debug 应为 False，实际 {getattr(default, 'debug', None)!r}"),
    ]
    for ok, msg in checks:
        if not ok:
            return False, msg

    # 自定义环境变量
    os.environ["APP_HOST"] = "h"
    os.environ["APP_PORT"] = "9001"
    os.environ["APP_DEBUG"] = "true"
    try:
        c = Config.from_env()
    except Exception as e:
        return False, f"Config.from_env() 自定义环境变量场景抛异常: {type(e).__name__}: {e}"
    if c.host != "h":
        return False, f"APP_HOST 未生效：{c.host!r}"
    if c.port != 9001 or not isinstance(c.port, int):
        return False, f"APP_PORT 应转成 int 9001，实际 {c.port!r}"
    if c.debug is not True:
        return False, f"APP_DEBUG='true' 应 → True，实际 {c.debug!r}"

    # debug=false 场景
    os.environ["APP_DEBUG"] = "0"
    try:
        c = Config.from_env()
    except Exception as e:
        return False, f"APP_DEBUG='0' 场景抛异常: {type(e).__name__}: {e}"
    if c.debug is not False:
        return False, f"APP_DEBUG='0' 应 → False，实际 {c.debug!r}"

    return True, "from_env 行为符合预期"


def main() -> int:
    module, err = _load_config()
    if err:
        _emit({"passed": False, "details": err})
        return 1

    Config = getattr(module, "Config", None)
    if Config is None:
        _emit({"passed": False, "details": "config.py 中找不到 Config 类"})
        return 1

    ok, detail = _check_from_env(Config)
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    test_file = _WORKSPACE / "test_config.py"
    if not test_file.is_file():
        _emit({
            "passed": False,
            "details": f"from_env 行为对，但缺测试文件：{test_file}",
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
        "details": f"pytest exit={proc.returncode}; from_env 行为校验通过",
    })
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
