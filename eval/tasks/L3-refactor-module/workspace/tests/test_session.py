"""session 本身的行为测试 —— 重构后应该从 app.session 导出."""

from __future__ import annotations

import time

import pytest

from app.session import (
    Session,
    SessionNotFound,
    create_session,
    expire_session,
    get_session,
    purge_expired,
)


def test_create_and_get_roundtrip() -> None:
    s = create_session(user_id=1)
    assert isinstance(s, Session)
    assert s.user_id == 1
    got = get_session(s.token)
    assert got is s


def test_get_unknown_raises() -> None:
    with pytest.raises(SessionNotFound):
        get_session("nope")


def test_expire_removes_session() -> None:
    s = create_session(user_id=1)
    expire_session(s.token)
    with pytest.raises(SessionNotFound):
        get_session(s.token)


def test_expire_unknown_is_noop() -> None:
    # 不应该因为 token 不存在就报错
    expire_session("never-existed")


def test_is_expired_respects_ttl() -> None:
    s = Session(token="t", user_id=1, created_at=0.0, ttl_seconds=10)
    assert s.is_expired(now=20.0) is True
    assert s.is_expired(now=5.0) is False


def test_purge_expired_cleans_only_old() -> None:
    fresh = create_session(user_id=1, ttl_seconds=1000)
    stale = create_session(user_id=2, ttl_seconds=1)
    # 手动把 stale 的 created_at 往回拨
    stale.created_at = time.time() - 10
    n = purge_expired()
    assert n == 1
    # 新的还在
    assert get_session(fresh.token).user_id == 1
    with pytest.raises(SessionNotFound):
        get_session(stale.token)
