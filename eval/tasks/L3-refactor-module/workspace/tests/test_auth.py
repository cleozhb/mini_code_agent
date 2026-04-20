"""auth 对外行为：login / logout / verify —— 签名和行为在重构前后都不变."""

from __future__ import annotations

import pytest

from app.auth import AuthError, Session, login, logout, verify


def test_login_success_returns_session() -> None:
    s = login(user_id=1, password="alice_pw")
    assert isinstance(s, Session)
    assert s.user_id == 1


def test_login_wrong_password_raises() -> None:
    with pytest.raises(AuthError):
        login(user_id=1, password="wrong")


def test_login_unknown_user_raises() -> None:
    with pytest.raises(AuthError):
        login(user_id=999, password="anything")


def test_verify_returns_user_id() -> None:
    s = login(user_id=2, password="bob_pw")
    assert verify(s.token) == 2


def test_logout_then_verify_fails() -> None:
    # 登出后 token 应该查不到（SessionNotFound 或 AuthError 都算合格，由实现决定）
    s = login(user_id=1, password="alice_pw")
    logout(s.token)
    with pytest.raises(Exception):  # noqa: B017
        verify(s.token)


def test_verify_expired_raises_auth_error() -> None:
    s = login(user_id=1, password="alice_pw")
    # 手动让它过期
    s.created_at -= s.ttl_seconds + 10
    with pytest.raises(AuthError):
        verify(s.token)
