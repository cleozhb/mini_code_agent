from __future__ import annotations

from user import User, collect_ids, find_by_id, format_user


def test_format_user():
    u = User(user_id=5, name="alice")
    assert format_user(u) == "User#5: alice"


def test_find_by_id_hit():
    users = [User(user_id=1, name="a"), User(user_id=2, name="b")]
    assert find_by_id(users, 2).name == "b"


def test_find_by_id_miss():
    users = [User(user_id=1, name="a")]
    assert find_by_id(users, 99) is None


def test_collect_ids():
    users = [User(user_id=10, name="x"), User(user_id=20, name="y")]
    assert collect_ids(users) == [10, 20]
