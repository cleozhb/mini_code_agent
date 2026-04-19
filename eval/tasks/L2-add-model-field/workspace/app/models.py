"""数据层：内存里存的 User 实体."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    """应用里存的 User 领域对象.

    当前只有 id / name 两个字段，业务侧想支持邮件通知，需要扩展。
    """

    id: int
    name: str
