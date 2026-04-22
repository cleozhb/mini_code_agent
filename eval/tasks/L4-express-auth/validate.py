#!/usr/bin/env python3
"""L4-express-auth 验证脚本.

协议（见 DESIGN.md §3 validate.py）：
- cwd 是该 run 的临时 workspace（runner 在外面 chdir 好）
- 可以打印任意调试信息，runner 只解析 **最后一行 stdout** 的 JSON
- JSON 必须包含 {"passed": bool, "details": str, ...}

判定逻辑：
1. 检查必需文件存在
2. app.js 引用了认证路由
3. middleware/auth.js 包含 JWT 验证逻辑
4. models/user.js 包含用户创建方法
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

    # 检查必需文件存在
    required_files = [
        "models/user.js",
        "routes/auth.js",
        "middleware/auth.js",
        "routes/protected.js",
        "app.js",
    ]
    for f in required_files:
        if not (_WORKSPACE / f).is_file():
            errors.append(f"缺少文件: {f}")

    if errors:
        _emit({"passed": False, "details": "; ".join(errors)})
        return 1

    # 检查 app.js 引用了认证路由
    app_content = (_WORKSPACE / "app.js").read_text()
    if "auth" not in app_content.lower():
        errors.append("app.js 未引用认证路由")

    # 检查 JWT 中间件包含关键逻辑
    auth_mw = (_WORKSPACE / "middleware/auth.js").read_text()
    if "verify" not in auth_mw.lower() and "jwt" not in auth_mw.lower():
        errors.append("middleware/auth.js 未包含 JWT 验证逻辑")

    # 检查用户模型包含基本方法
    user_model = (_WORKSPACE / "models/user.js").read_text()
    if "create" not in user_model.lower() and "register" not in user_model.lower():
        errors.append("models/user.js 未包含用户创建方法")

    if errors:
        _emit({"passed": False, "details": "; ".join(errors)})
        return 1

    _emit({"passed": True, "details": "所有检查通过"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
