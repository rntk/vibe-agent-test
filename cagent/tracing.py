"""Small dependency-free tracing helpers."""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, is_dataclass
from hashlib import sha256
from html import escape
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
        self.record.attributes[key] = self.tracer.to_trace_value(value)


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
        self._value_refs: dict[str, str] = {}
        self._values_by_ref: dict[str, Any] = {}
        self._next_value_ref = 1

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
            attributes=self.to_trace_dict(attributes or {}),
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
            "deduplication": {
                "value_count": len(self._value_refs),
                "values": self._values_by_ref,
                "ref_format": (
                    "Duplicate attribute values are replaced by "
                    "{'$ref': 'trace_value_N', '$deduplicated': true}."
                ),
            },
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

    def to_trace_dict(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Return trace attributes with repeated container values tombstoned."""

        return {key: self.to_trace_value(value) for key, value in values.items()}

    def to_trace_value(self, value: Any) -> Any:
        """Return a JSON-safe trace value, replacing duplicates with references."""

        return self._deduplicate_value(_to_jsonable(value))

    def _deduplicate_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            deduped_mapping = {
                str(key): self._deduplicate_value(item) for key, item in value.items()
            }
            return self._deduplicate_container(deduped_mapping)
        if isinstance(value, list):
            deduped_list = [self._deduplicate_value(item) for item in value]
            return self._deduplicate_container(deduped_list)
        return value

    def _deduplicate_container(self, value: dict[str, Any] | list[Any]) -> Any:
        if not value:
            return value

        fingerprint = _trace_value_fingerprint(value)
        if fingerprint in self._value_refs:
            return {
                "$ref": self._value_refs[fingerprint],
                "$deduplicated": True,
            }

        ref = f"trace_value_{self._next_value_ref}"
        self._next_value_ref += 1
        self._value_refs[fingerprint] = ref
        self._values_by_ref[ref] = value
        return value


def write_trace_html(
    trace_file_path: str | Path,
    html_file_path: str | Path | None = None,
) -> Path:
    """Render a readable HTML conversation view for a JSON trace file."""

    trace_path = Path(trace_file_path)
    output_path = Path(html_file_path) if html_file_path else trace_path.with_suffix(
        ".html"
    )
    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    html = render_trace_html(trace_data, trace_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_trace_html(trace_data: Mapping[str, Any], trace_name: str = "trace") -> str:
    """Return a standalone HTML page for a trace document."""

    value_refs = _trace_value_refs(trace_data)
    spans = _flatten_spans(trace_data.get("spans", []))
    conversation = _conversation_events(spans, value_refs)
    span_count = trace_data.get("span_count", len(spans))

    conversation_html = "\n".join(
        _render_conversation_event(event) for event in conversation
    )
    if not conversation_html:
        conversation_html = (
            '<p class="empty">No LLM conversation messages were found in this '
            "trace.</p>"
        )

    timeline_html = "\n".join(_render_span_summary(span, value_refs) for span in spans)
    if not timeline_html:
        timeline_html = '<p class="empty">No spans were found in this trace.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(trace_name)} Trace</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #667085;
      --line: #d9dee7;
      --system: #596579;
      --user: #0f766e;
      --assistant: #334155;
      --tool: #9a3412;
      --error: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 20px clamp(16px, 4vw, 48px);
    }}
    h1, h2 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    .meta {{ color: var(--muted); margin-top: 6px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
      gap: 20px;
      padding: 20px clamp(16px, 4vw, 48px) 40px;
    }}
    section {{
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .conversation {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .message {{
      border-left: 4px solid var(--assistant);
      background: #fbfcfe;
      border-radius: 6px;
      padding: 12px;
    }}
    .message.system {{ border-color: var(--system); }}
    .message.user {{ border-color: var(--user); }}
    .message.assistant {{ border-color: var(--assistant); }}
    .message.tool {{ border-color: var(--tool); }}
    .role {{
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    details {{
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    summary {{ cursor: pointer; color: var(--muted); }}
    .timeline {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .span {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
    }}
    .span.error {{ border-color: var(--error); }}
    .span-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-weight: 700;
    }}
    .span small {{ color: var(--muted); }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(trace_name)} Trace</h1>
    <div class="meta">{escape(str(span_count))} spans captured</div>
  </header>
  <main>
    <section>
      <h2>Conversation</h2>
      <div class="conversation">
        {conversation_html}
      </div>
    </section>
    <section>
      <h2>Span Timeline</h2>
      <div class="timeline">
        {timeline_html}
      </div>
    </section>
  </main>
</body>
</html>
"""


def get_trace() -> Trace | NoOpTrace:
    """Return the current trace collector."""

    return _CURRENT_TRACE.get()


def set_trace(trace: Trace | NoOpTrace) -> Token[Trace | NoOpTrace]:
    """Set the current trace collector."""

    return _CURRENT_TRACE.set(trace)


def reset_trace(token: Token[Trace | NoOpTrace]) -> None:
    """Restore a previous trace collector."""

    _CURRENT_TRACE.reset(token)


def _trace_value_refs(trace_data: Mapping[str, Any]) -> Mapping[str, Any]:
    deduplication = trace_data.get("deduplication")
    if not isinstance(deduplication, Mapping):
        return {}
    values = deduplication.get("values")
    if not isinstance(values, Mapping):
        return {}
    return values


def _resolve_trace_refs(value: Any, value_refs: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if value.get("$deduplicated") is True and isinstance(ref, str):
            resolved = value_refs.get(ref)
            if resolved is not None:
                return _resolve_trace_refs(resolved, value_refs)
        return {
            str(key): _resolve_trace_refs(item, value_refs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_trace_refs(item, value_refs) for item in value]
    return value


def _flatten_spans(spans: Any) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    if not isinstance(spans, list):
        return flattened

    for span in spans:
        if not isinstance(span, Mapping):
            continue
        span_dict = dict(span)
        flattened.append(span_dict)
        flattened.extend(_flatten_spans(span.get("children", [])))
    return sorted(flattened, key=lambda item: item.get("started_at") or 0)


def _conversation_events(
    spans: Sequence[Mapping[str, Any]],
    value_refs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    displayed_message_count = 0

    for span in spans:
        if span.get("name") != "llm.complete":
            continue

        attributes = _resolve_trace_refs(span.get("attributes", {}), value_refs)
        if not isinstance(attributes, Mapping):
            continue

        messages = attributes.get("all_messages", [])
        if isinstance(messages, list):
            for message in messages[displayed_message_count:]:
                if isinstance(message, Mapping):
                    events.append(dict(message))
            displayed_message_count = max(displayed_message_count, len(messages))

        response = attributes.get("response")
        response_message = _response_to_message(response)
        if response_message is not None:
            events.append(response_message)
            displayed_message_count += 1

    return events


def _response_to_message(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, Mapping):
        return None

    content = response.get("content")
    reasoning = response.get("reasoning")
    tool_calls = response.get("tool_calls")
    has_content = content is not None and content != ""
    has_reasoning = reasoning is not None and reasoning != ""
    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    if not has_content and not has_reasoning and not has_tool_calls:
        return None

    return {
        "role": "assistant",
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls or [],
    }


def _render_conversation_event(event: Mapping[str, Any]) -> str:
    role = str(event.get("role") or "message")
    role_class = role if role in {"system", "user", "assistant", "tool"} else "message"
    tool_call_id = event.get("tool_call_id")
    meta = f"tool call {tool_call_id}" if tool_call_id else ""
    content = event.get("content")
    if content is None:
        content_text = ""
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = json.dumps(content, indent=2, sort_keys=True)

    tool_calls = event.get("tool_calls")
    tool_calls_html = ""
    if isinstance(tool_calls, list) and tool_calls:
        tool_calls_html = (
            "<details open><summary>Tool calls</summary>"
            f"{_json_pre(tool_calls)}</details>"
        )

    reasoning = event.get("reasoning")
    reasoning_html = ""
    if isinstance(reasoning, str) and reasoning:
        reasoning_html = (
            "<details open><summary>Reasoning</summary>"
            f"<pre>{escape(reasoning)}</pre></details>"
        )

    content_html = f"<pre>{escape(content_text)}</pre>" if content_text else ""
    return (
        f'<article class="message {escape(role_class)}">'
        f'<div class="role"><span>{escape(role)}</span>'
        f"<span>{escape(meta)}</span></div>"
        f"{reasoning_html}{content_html}{tool_calls_html}</article>"
    )


def _render_span_summary(
    span: Mapping[str, Any],
    value_refs: Mapping[str, Any],
) -> str:
    name = str(span.get("name") or "span")
    status = str(span.get("status") or "ok")
    duration = span.get("duration_ms")
    duration_text = f"{duration:.1f} ms" if isinstance(duration, int | float) else ""
    attributes = _resolve_trace_refs(span.get("attributes", {}), value_refs)
    summary = _span_attribute_summary(attributes)
    status_class = " error" if status == "error" else ""
    error_html = ""
    if span.get("error"):
        error_html = f"<pre>{escape(str(span.get('error')))}</pre>"
    details_html = ""
    if attributes:
        details_html = (
            f"<details><summary>Attributes</summary>{_json_pre(attributes)}</details>"
        )

    return (
        f'<div class="span{status_class}">'
        f'<div class="span-title"><span>{escape(name)}</span>'
        f"<small>{escape(status)} {escape(duration_text)}</small></div>"
        f"<small>{escape(summary)}</small>{error_html}{details_html}</div>"
    )


def _span_attribute_summary(attributes: Any) -> str:
    if not isinstance(attributes, Mapping):
        return ""
    parts: list[str] = []
    tool_name = attributes.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")
    message_count = attributes.get("message_count")
    if message_count:
        parts.append(f"messages={message_count}")
    response_content = attributes.get("response_content")
    if isinstance(response_content, str) and response_content:
        parts.append(response_content[:80])
    response_reasoning = attributes.get("response_reasoning")
    if isinstance(response_reasoning, str) and response_reasoning:
        parts.append(f"reasoning={response_reasoning[:80]}")
    return " | ".join(parts)


def _json_pre(value: Any) -> str:
    rendered = json.dumps(value, indent=2, sort_keys=True)
    return f"<pre>{escape(rendered)}</pre>"


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


def _trace_value_fingerprint(value: dict[str, Any] | list[Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


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
