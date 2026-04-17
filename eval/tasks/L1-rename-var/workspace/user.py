"""用户数据模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    user_id: int
    name: str


def format_user(u: User) -> str:
    return f"User#{u.user_id}: {u.name}"


def find_by_id(users: list[User], user_id: int) -> User | None:
    for candidate in users:
        if candidate.user_id == user_id:
            return candidate
    return None


def collect_ids(users: list[User]) -> list[int]:
    return [u.user_id for u in users]
