"""全量 Agent / LLM trace 记录器."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .serializer import TraceSerializer

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1


@dataclass
class TraceContext:
    """一次 trace session 的上下文."""

    session_id: str
    started_at: str
    project_dir: str
    provider: str
    model: str
    session_dir: Path
    trace_schema_version: int = TRACE_SCHEMA_VERSION
    llm_call_index: int = 0
    event_index: int = 0


class TraceRecorder:
    """把 Agent 运行过程写入 append-only trace 文件."""

    def __init__(
        self,
        *,
        project_dir: Path,
        provider: str,
        model: str,
        session_id: str | None = None,
        started_at: datetime | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        now = started_at or datetime.now(timezone.utc)
        sid = session_id or uuid.uuid4().hex[:6]
        stamp = now.astimezone().strftime("%Y%m%d-%H%M%S")
        session_dir = project_dir / ".agent" / "traces" / f"{stamp}-{sid}"

        self.serializer = TraceSerializer()
        self.context = TraceContext(
            session_id=sid,
            started_at=now.isoformat(timespec="seconds"),
            project_dir=str(project_dir),
            provider=provider,
            model=model,
            session_dir=session_dir,
        )
        self.llm_dir = session_dir / "llm"
        self.events_path = session_dir / "events.jsonl"
        self.on_error = on_error
        self._write_failed = False
        self._finished = False

        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            session_dir / "session.json",
            {
                "session_id": self.context.session_id,
                "started_at": self.context.started_at,
                "project_dir": self.context.project_dir,
                "provider": self.context.provider,
                "model": self.context.model,
                "trace_schema_version": self.context.trace_schema_version,
            },
        )
        self.record_event("session_start", {
            "session_id": self.context.session_id,
            "project_dir": self.context.project_dir,
            "provider": self.context.provider,
            "model": self.context.model,
        })

    @property
    def session_dir(self) -> Path:
        """当前 trace session 目录."""
        return self.context.session_dir

    def next_llm_call_id(self) -> str:
        """分配一个递增的 LLM call id."""
        self.context.llm_call_index += 1
        return f"{self.context.llm_call_index:04d}"

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """向 events.jsonl 追加一条事件."""
        self.context.event_index += 1
        event = {
            "event_id": self.context.event_index,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "payload": payload or {},
        }
        self._append_jsonl(self.events_path, event)

    def start_agent_run(self, user_message: str) -> str:
        """记录一次 Agent run 开始."""
        run_id = uuid.uuid4().hex[:8]
        self.record_event("agent_run_start", {
            "run_id": run_id,
            "user_message": user_message,
        })
        return run_id

    def finish_agent_run(
        self,
        run_id: str | None,
        *,
        result: Any = None,
        error: BaseException | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """记录一次 Agent run 结束."""
        payload: dict[str, Any] = {"run_id": run_id}
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        if stop_reason is not None:
            payload["stop_reason"] = stop_reason
        self.record_event("agent_run_finish", payload)

    def start_llm_call(
        self,
        *,
        model: str,
        backend: str,
        mode: str,
        messages: Any,
        tools: Any,
        response_format: Any,
    ) -> str:
        """记录 LLM normalized request，并返回 call_id."""
        call_id = self.next_llm_call_id()
        payload = {
            "call_id": call_id,
            "model": model,
            "backend": backend,
            "mode": mode,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        self.write_llm_json(call_id, "request.normalized", payload)
        self.record_event("llm_call_start", {
            "call_id": call_id,
            "model": model,
            "backend": backend,
            "mode": mode,
            "normalized_request": f"llm/{call_id}-request.normalized.json",
        })
        return call_id

    def record_llm_provider_request(self, call_id: str | None, payload: Any) -> None:
        """记录 provider SDK 调用前的最终 payload."""
        if not call_id:
            return
        self.write_llm_json(call_id, "request.provider", payload)

    def record_llm_raw_response(
        self,
        call_id: str | None,
        response: Any,
        *,
        label: str = "response.raw",
    ) -> None:
        """记录非流式 raw response 或额外 raw 对象."""
        if not call_id:
            return
        self.write_llm_json(call_id, label, response)

    def record_llm_stream_event(self, call_id: str | None, event: Any) -> None:
        """记录 streaming raw chunk/event."""
        if not call_id:
            return
        path = self.llm_dir / f"{call_id}-stream.events.jsonl"
        self._append_jsonl(path, {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
        })

    def record_llm_parsed_response(self, call_id: str | None, response: Any) -> None:
        """记录统一 LLMResponse 解析结果."""
        if not call_id:
            return
        self.write_llm_json(call_id, "response.parsed", response)

    def finish_llm_call(
        self,
        call_id: str | None,
        *,
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """记录 LLM 调用结束事件."""
        if not call_id:
            return
        payload: dict[str, Any] = {"call_id": call_id}
        if response is not None:
            payload["usage"] = getattr(response, "usage", None)
            payload["tool_calls"] = getattr(response, "tool_calls", [])
            payload["content"] = getattr(response, "content", "")
            payload["parsed_response"] = f"llm/{call_id}-response.parsed.json"
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self.record_event("llm_call_finish", payload)

    def record_tool_call_start(
        self,
        *,
        llm_call_id: str | None,
        tool_call: Any,
    ) -> None:
        """记录工具调用开始."""
        self.record_event("tool_call_start", {
            "llm_call_id": llm_call_id,
            "tool_call_id": getattr(tool_call, "id", ""),
            "tool_name": getattr(tool_call, "name", ""),
            "arguments": getattr(tool_call, "arguments", {}),
            "raw_arguments": getattr(tool_call, "raw_arguments", ""),
            "parse_error": getattr(tool_call, "parse_error", None),
        })

    def record_tool_call_result(
        self,
        *,
        llm_call_id: str | None,
        tool_call: Any,
        result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录工具调用结果."""
        self.record_event("tool_call_result", {
            "llm_call_id": llm_call_id,
            "tool_call_id": getattr(tool_call, "id", ""),
            "tool_name": getattr(tool_call, "name", ""),
            "arguments": getattr(tool_call, "arguments", {}),
            "result": result,
            "metadata": metadata or {},
        })

    def write_llm_json(self, call_id: str, label: str, payload: Any) -> None:
        """写入一个 LLM 调用相关 JSON 文件."""
        self._write_json(self.llm_dir / f"{call_id}-{label}.json", payload)

    def finish_session(self) -> None:
        """记录 session 结束."""
        if self._finished:
            return
        self._finished = True
        self.record_event("session_finish", {
            "session_id": self.context.session_id,
        })

    def _write_json(self, path: Path, payload: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    self.serializer.to_jsonable(payload),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001 - trace 不能影响主流程
            self._handle_write_error(e)

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        self.serializer.to_jsonable(payload),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:  # noqa: BLE001 - trace 不能影响主流程
            self._handle_write_error(e)

    def _handle_write_error(self, error: BaseException) -> None:
        if self._write_failed:
            logger.warning("trace 写入失败: %s", error)
            return
        self._write_failed = True
        logger.warning("trace 写入失败: %s", error)
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception:
                logger.debug("trace on_error callback failed", exc_info=True)
        try:
            self.record_event("trace_error", {
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            })
        except Exception:
            logger.debug("trace_error 事件写入也失败", exc_info=True)
