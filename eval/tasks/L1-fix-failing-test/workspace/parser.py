"""Key-value 格式解析器。输入形如

    foo=1
    bar=http://example.com?x=1

解析成 dict。支持 '#' 开头的注释行和空行。
"""

from __future__ import annotations


def parse_key_value_lines(text: str) -> dict[str, str]:
    """解析 k=v 多行文本。空行和 # 开头的注释行会被忽略。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("=")
        key, value = parts[0], parts[1]
        out[key] = value
    return out
