"""User 相关的单元测试.

这批测试编码了"User 应该有 email 字段"这一期望行为，以及原来 id/name 相关
的行为必须**保留不变**。初始代码里还没加 email，所以这些 test 一开始是失败的 ——
这是刻意的：把 tests 当作规约，改动完成后应全部变绿。
"""

from __future__ import annotations

import pytest

from app.api import (
    UserAlreadyExists,
    UserNotFound,
    create_user,
    get_user,
)
from app.models import User
from app.schemas import UserCreateIn, UserOut


# ---- 新增字段相关：初始状态下这几条会失败 ------------------------------------


def test_user_model_accepts_email_kwarg():
    u = User(id=1, name="Alice", email="alice@example.com")
    assert u.email == "alice@example.com"


def test_usercreatein_accepts_email_kwarg():
    payload = UserCreateIn(id=7, name="Bob", email="bob@example.com")
    assert payload.email == "bob@example.com"


def test_userout_accepts_email_kwarg():
    out = UserOut(id=7, name="Bob", email="bob@example.com")
    assert out.email == "bob@example.com"


def test_email_is_required_on_create_in():
    # email 是必填 —— 没默认值，dataclass 缺参就该抛 TypeError
    with pytest.raises(TypeError):
        UserCreateIn(id=1, name="Alice")  # type: ignore[call-arg]


def test_create_and_get_preserves_email():
    out = create_user(UserCreateIn(id=42, name="Carol", email="carol@ex.com"))
    assert isinstance(out, UserOut)
    assert out.id == 42
    assert out.name == "Carol"
    assert out.email == "carol@ex.com"

    got = get_user(42)
    assert got.id == 42
    assert got.name == "Carol"
    assert got.email == "carol@ex.com"


def test_create_two_distinct_users_keep_separate_emails():
    create_user(UserCreateIn(id=1, name="A", email="a@x.com"))
    create_user(UserCreateIn(id=2, name="B", email="b@x.com"))
    assert get_user(1).email == "a@x.com"
    assert get_user(2).email == "b@x.com"


# ---- 回归：原先的 id/name + 错误路径必须保持不变 -----------------------------


def test_duplicate_id_raises_already_exists():
    create_user(UserCreateIn(id=1, name="A", email="a@x.com"))
    with pytest.raises(UserAlreadyExists):
        create_user(UserCreateIn(id=1, name="A2", email="a2@x.com"))


def test_get_missing_raises_not_found():
    with pytest.raises(UserNotFound):
        get_user(999)


def test_name_still_works_after_change():
    # 确保 name 字段没被意外破坏
    create_user(UserCreateIn(id=3, name="Zoe", email="z@x.com"))
    assert get_user(3).name == "Zoe"
