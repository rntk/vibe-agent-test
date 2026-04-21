"""Small dependency-free tracing helpers."""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass(slots=True)
class SpanRecord:
    """Collected data for one traced operation."""

    id: str
    name: str
    parent_id: str | None
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list[SpanRecord] = field(default_factory=list)


class Span(Protocol):
    """Mutable handle for the active span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute to this span."""


@dataclass(slots=True)
class _ActiveSpan:
    record: SpanRecord
    tracer: Trace
    token: Token[_ActiveSpan | None] | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.record.attributes[key] = _to_jsonable(value)


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


class NoOpTrace:
    """Trace implementation used when tracing is disabled."""

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        yield _NoOpSpan()

    def flush(self, file_path: str | Path | None = None) -> None:
        return None


_CURRENT_TRACE: ContextVar[Trace | NoOpTrace] = ContextVar(
    "cagent_current_trace",
    default=NoOpTrace(),
)
_CURRENT_SPAN: ContextVar[_ActiveSpan | None] = ContextVar(
    "cagent_current_span",
    default=None,
)


class Trace:
    """Collect parent/child spans and emit a JSON trace."""

    def __init__(self) -> None:
        self.roots: list[SpanRecord] = []
        self.spans: list[SpanRecord] = []

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent = _CURRENT_SPAN.get()
        record = SpanRecord(
            id=uuid4().hex,
            name=name,
            parent_id=parent.record.id if parent else None,
            started_at=time.time(),
            attributes=_to_jsonable_dict(attributes or {}),
        )
        self.spans.append(record)
        if parent is None:
            self.roots.append(record)
        else:
            parent.record.children.append(record)

        active_span = _ActiveSpan(record=record, tracer=self)
        active_span.token = _CURRENT_SPAN.set(active_span)
        try:
            yield active_span
        except Exception as exc:
            record.status = "error"
            record.error = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            raise
        finally:
            record.ended_at = time.time()
            record.duration_ms = (record.ended_at - record.started_at) * 1000
            if active_span.token is not None:
                _CURRENT_SPAN.reset(active_span.token)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace document."""

        return {
            "span_count": len(self.spans),
            "spans": [_span_to_dict(span) for span in self.roots],
        }

    def to_json(self) -> str:
        """Return the collected trace as formatted JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def flush(self, file_path: str | Path | None = None) -> None:
        """Log the trace and optionally write it to a file."""

        rendered = self.to_json()
        logging.info("Trace result:\n%s", rendered)
        if file_path:
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")


def get_trace() -> Trace | NoOpTrace:
    """Return the current trace collector."""

    return _CURRENT_TRACE.get()


def set_trace(trace: Trace | NoOpTrace) -> Token[Trace | NoOpTrace]:
    """Set the current trace collector."""

    return _CURRENT_TRACE.set(trace)


def reset_trace(token: Token[Trace | NoOpTrace]) -> None:
    """Restore a previous trace collector."""

    _CURRENT_TRACE.reset(token)


def _span_to_dict(span: SpanRecord) -> dict[str, Any]:
    return {
        "id": span.id,
        "name": span.name,
        "parent_id": span.parent_id,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "duration_ms": span.duration_ms,
        "status": span.status,
        "attributes": span.attributes,
        "error": span.error,
        "children": [_span_to_dict(child) for child in span.children],
    }


def _to_jsonable_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _to_jsonable(value) for key, value in values.items()}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_to_jsonable(item) for item in value]
    if is_dataclass(value):
        return _to_jsonable(
            {
                key: getattr(value, key)
                for key in getattr(value, "__dataclass_fields__", {})
            }
        )
    return repr(value)
