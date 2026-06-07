"""Trace 序列化工具."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - pydantic 是项目依赖，兜底避免导入期炸裂
    BaseModel = None  # type: ignore[assignment]


class TraceSerializer:
    """把各种运行期对象转换为 JSON 友好的结构."""

    def to_jsonable(self, value: Any) -> Any:
        """转换为可被 json.dump 写入的对象."""
        return self._to_jsonable(value, seen=set())

    def _to_jsonable(self, value: Any, seen: set[int]) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return repr(value)

        obj_id = id(value)
        if obj_id in seen:
            return repr(value)

        if isinstance(value, dict):
            seen.add(obj_id)
            try:
                return {
                    str(self._to_jsonable(k, seen)): self._to_jsonable(v, seen)
                    for k, v in value.items()
                }
            finally:
                seen.discard(obj_id)

        if isinstance(value, list | tuple | set | frozenset):
            seen.add(obj_id)
            try:
                return [self._to_jsonable(item, seen) for item in value]
            finally:
                seen.discard(obj_id)

        if BaseModel is not None and isinstance(value, BaseModel):
            try:
                return self._to_jsonable(value.model_dump(mode="json"), seen)
            except Exception:
                return repr(value)

        if is_dataclass(value) and not isinstance(value, type):
            seen.add(obj_id)
            try:
                return {
                    f.name: self._to_jsonable(getattr(value, f.name), seen)
                    for f in fields(value)
                }
            finally:
                seen.discard(obj_id)

        if isinstance(value, SimpleNamespace):
            return self._to_jsonable(vars(value), seen)

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self._to_jsonable(model_dump(mode="json"), seen)
            except TypeError:
                try:
                    return self._to_jsonable(model_dump(), seen)
                except Exception:
                    return repr(value)
            except Exception:
                return repr(value)

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return self._to_jsonable(to_dict(), seen)
            except Exception:
                return repr(value)

        if hasattr(value, "__dict__"):
            seen.add(obj_id)
            try:
                return self._to_jsonable(vars(value), seen)
            except Exception:
                return repr(value)
            finally:
                seen.discard(obj_id)

        return repr(value)
