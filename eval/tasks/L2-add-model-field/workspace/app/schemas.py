"""传输层：API 入参 / 出参 schema.

`UserCreateIn` 是 POST /users 的请求体；`UserOut` 是 GET /users/{id} 的响应体。
这里只放 schema，不关心存储。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserCreateIn:
    """创建 User 的入参."""

    id: int
    name: str


@dataclass
class UserOut:
    """返回给客户端的 User 视图."""

    id: int
    name: str
