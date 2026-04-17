#!/usr/bin/env python3
"""L1-rename-var 的验证脚本.

协议同 DESIGN.md §3。判定逻辑：
1. user.py 和 tests/test_user.py 的源码里完全不出现 `user_id` 这个标识符
2. 加载 user.py 后，能用 uid 关键字构造 User：User(uid=5, name="x")
3. 实例上有 .uid 属性、没有 .user_id 属性
4. find_by_id / collect_ids 等公共函数保留，行为不变
5. pytest 全部通过
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path.cwd()
_TARGET_FILES = ("user.py", "tests/test_user.py")
_USER_ID_PATTERN = re.compile(r"\buser_id\b")


def _emit(payload: dict) -> None:
    print()
    print(json.dumps(payload, ensure_ascii=False))


def _load_user_module():
    path = _WORKSPACE / "user.py"
    if not path.is_file():
        return None, f"找不到 user.py：{path}"
    spec = importlib.util.spec_from_file_location("user_under_eval", path)
    if spec is None or spec.loader is None:
        return None, "无法构造 user.py 的 import spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"执行 user.py 失败: {type(e).__name__}: {e}"
    return module, None


def _check_no_user_id_identifier() -> tuple[bool, str]:
    for rel in _TARGET_FILES:
        path = _WORKSPACE / rel
        if not path.is_file():
            return False, f"找不到目标文件 {rel}"
        content = path.read_text(encoding="utf-8")
        m = _USER_ID_PATTERN.search(content)
        if m:
            # 定位行号方便调试
            line_no = content.count("\n", 0, m.start()) + 1
            return False, f"{rel} 第 {line_no} 行仍含 user_id 标识符"
    return True, "源文件中不再有 user_id"


def _check_renamed_behavior(module) -> tuple[bool, str]:
    User = getattr(module, "User", None)
    if User is None:
        return False, "user.py 中找不到 User 类"
    try:
        u = User(uid=5, name="alice")
    except TypeError as e:
        return False, f"User(uid=..., name=...) 构造失败: {e}"
    if not hasattr(u, "uid") or u.uid != 5:
        return False, f"实例缺少正确的 uid 属性，实际 uid={getattr(u, 'uid', None)!r}"
    if hasattr(u, "user_id"):
        return False, "实例上仍残留 user_id 属性（不应保留 alias）"
    if getattr(u, "name", None) != "alice":
        return False, f"name 字段异常：{getattr(u, 'name', None)!r}"

    # find_by_id / collect_ids 需要保留
    find_by_id = getattr(module, "find_by_id", None)
    collect_ids = getattr(module, "collect_ids", None)
    format_user = getattr(module, "format_user", None)
    if find_by_id is None or collect_ids is None or format_user is None:
        return False, "缺失公共函数 find_by_id / collect_ids / format_user"

    users = [User(uid=1, name="a"), User(uid=2, name="b")]
    if find_by_id(users, 2) is None or find_by_id(users, 2).name != "b":
        return False, "find_by_id 行为不符"
    if collect_ids(users) != [1, 2]:
        return False, f"collect_ids 行为不符：{collect_ids(users)!r}"
    return True, "重命名后行为符合预期"


def main() -> int:
    ok, detail = _check_no_user_id_identifier()
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    module, err = _load_user_module()
    if err:
        _emit({"passed": False, "details": err})
        return 1

    ok, detail = _check_renamed_behavior(module)
    if not ok:
        _emit({"passed": False, "details": detail})
        return 1

    test_file = _WORKSPACE / "tests" / "test_user.py"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-x", "-q", "--no-header"],
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
        "details": f"pytest exit={proc.returncode}; rename check OK",
    })
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
