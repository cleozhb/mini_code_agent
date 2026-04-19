"""API 层：负责 schema ↔ model 之间的转换和内存存储.

这是一个非常小的"伪 web 层"：没有真实框架，就是几个函数供测试直接调用。
"""

from __future__ import annotations

from .models import User
from .schemas import UserCreateIn, UserOut

# 内存存储，简单映射 id → User
_USERS: dict[int, User] = {}


class UserNotFound(Exception):
    """get_user 找不到给定 id 时抛这个，而不是返回 None."""


class UserAlreadyExists(Exception):
    """create_user 碰到重复 id 时抛，而不是静默覆盖."""


def create_user(data: UserCreateIn) -> UserOut:
    """创建一个新用户；id 重复抛 UserAlreadyExists."""
    if data.id in _USERS:
        raise UserAlreadyExists(f"user id already exists: {data.id}")
    user = User(id=data.id, name=data.name)
    _USERS[data.id] = user
    return UserOut(id=user.id, name=user.name)


def get_user(uid: int) -> UserOut:
    """按 id 取用户；不存在抛 UserNotFound."""
    user = _USERS.get(uid)
    if user is None:
        raise UserNotFound(f"user not found: {uid}")
    return UserOut(id=user.id, name=user.name)


def reset_store() -> None:
    """仅供测试清空内存存储."""
    _USERS.clear()
