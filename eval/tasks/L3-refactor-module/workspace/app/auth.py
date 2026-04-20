"""登录认证 + 会话管理（目前都塞在这一个文件里，需要拆分）."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Session 管理（目标：抽到 app/session.py）
# ---------------------------------------------------------------------------


@dataclass
class Session:
    token: str
    user_id: int
    created_at: float
    ttl_seconds: int = 3600

    def is_expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now - self.created_at > self.ttl_seconds


_SESSIONS: dict[str, Session] = {}


class SessionNotFound(Exception):
    """查询/过期的 token 对应的 Session 不存在."""


def create_session(user_id: int, ttl_seconds: int = 3600) -> Session:
    token = secrets.token_hex(16)
    s = Session(
        token=token,
        user_id=user_id,
        created_at=time.time(),
        ttl_seconds=ttl_seconds,
    )
    _SESSIONS[token] = s
    return s


def get_session(token: str) -> Session:
    s = _SESSIONS.get(token)
    if s is None:
        raise SessionNotFound(token)
    return s


def expire_session(token: str) -> None:
    _SESSIONS.pop(token, None)


def purge_expired(now: float | None = None) -> int:
    now = time.time() if now is None else now
    dead = [t for t, s in _SESSIONS.items() if s.is_expired(now)]
    for t in dead:
        _SESSIONS.pop(t, None)
    return len(dead)


def reset_sessions() -> None:
    _SESSIONS.clear()


# ---------------------------------------------------------------------------
# Auth（登录/登出/校验）
# ---------------------------------------------------------------------------


# 真实项目会从 DB 查；这里硬编码两个测试用户
_USERS: dict[int, str] = {1: "alice_pw", 2: "bob_pw"}


class AuthError(Exception):
    """登录凭据错误 / session 过期 等认证类错误."""


def login(user_id: int, password: str) -> Session:
    if _USERS.get(user_id) != password:
        raise AuthError("bad credentials")
    return create_session(user_id)


def logout(token: str) -> None:
    expire_session(token)


def verify(token: str) -> int:
    """返回 token 对应的 user_id；无效抛 SessionNotFound、过期抛 AuthError."""
    s = get_session(token)
    if s.is_expired():
        expire_session(token)
        raise AuthError("session expired")
    return s.user_id
