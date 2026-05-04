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
        """Return JSON-safe trace attributes without changing their shape."""

        return {key: self.to_trace_value(value) for key, value in values.items()}

    def to_trace_value(self, value: Any) -> Any:
        """Return a JSON-safe trace value."""

        return _to_jsonable(value)


def write_trace_html(
    trace_file_path: str | Path,
    html_file_path: str | Path | None = None,
) -> Path:
    """Render a readable HTML conversation view for a JSON trace file."""

    trace_path = Path(trace_file_path)
    output_path = (
        Path(html_file_path) if html_file_path else trace_path.with_suffix(".html")
    )
    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    html = render_trace_html(trace_data, trace_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_trace_html(trace_data: Mapping[str, Any], trace_name: str = "trace") -> str:
    """Return a standalone HTML page for a trace document."""

    spans = _flatten_spans(trace_data.get("spans", []))
    report_spans = _deduplicate_report_spans(spans)
    conversation = _conversation_events(spans)
    span_count = trace_data.get("span_count", len(spans))

    conversation_html = "\n".join(
        _render_conversation_event(event) for event in conversation
    )
    if not conversation_html:
        conversation_html = (
            '<p class="empty">No LLM conversation messages were found in this '
            "trace.</p>"
        )

    timeline_html = "\n".join(_render_span_summary(span) for span in report_spans)
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
      --ctx-agent: #0f766e;
      --ctx-advisor: #7c3aed;
      --ctx-advisor_tool: #a21caf;
      --ctx-tool: #9a3412;
      --ctx-compaction: #0369a1;
      --ctx-summary: #64748b;
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
    .context-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .05em;
      text-transform: uppercase;
      padding: 1px 6px;
      border-radius: 3px;
      margin-left: 6px;
    }}
    .context-badge.agent {{ background: var(--ctx-agent); color: #fff; }}
    .context-badge.advisor {{ background: var(--ctx-advisor); color: #fff; }}
    .context-badge.advisor_tool {{ background: var(--ctx-advisor_tool); color: #fff; }}
    .context-badge.tool {{ background: var(--ctx-tool); color: #fff; }}
    .context-badge.compaction {{ background: var(--ctx-compaction); color: #fff; }}
    .context-badge.summary {{ background: var(--ctx-summary); color: #fff; }}
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
    .span.context-agent {{ border-left: 4px solid var(--ctx-agent); }}
    .span.context-advisor {{ border-left: 4px solid var(--ctx-advisor); }}
    .span.context-advisor_tool {{ border-left: 4px solid var(--ctx-advisor_tool); }}
    .span.context-tool {{ border-left: 4px solid var(--ctx-tool); }}
    .span.context-compaction {{ border-left: 4px solid var(--ctx-compaction); }}
    .span.context-summary {{ border-left: 4px solid var(--ctx-summary); }}
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


def _deduplicate_report_spans(
    spans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return spans with repeated containers compacted for HTML rendering only."""

    seen: dict[str, str] = {}
    next_ref = 1
    deduplicated_spans: list[dict[str, Any]] = []

    for span in spans:
        span_dict = dict(span)
        attributes = span.get("attributes", {})
        if isinstance(attributes, Mapping):
            attributes, next_ref = _deduplicate_report_value(
                attributes,
                seen,
                next_ref,
            )
        span_dict["attributes"] = attributes
        deduplicated_spans.append(span_dict)

    return deduplicated_spans


def _deduplicate_report_value(
    value: Any,
    seen: dict[str, str],
    next_ref: int,
) -> tuple[Any, int]:
    if isinstance(value, Mapping):
        deduplicated_mapping: dict[str, Any] = {}
        for key, item in value.items():
            deduplicated_item, next_ref = _deduplicate_report_value(
                item, seen, next_ref
            )
            deduplicated_mapping[str(key)] = deduplicated_item
        return _deduplicate_report_container(
            deduplicated_mapping,
            seen,
            next_ref,
        )

    if isinstance(value, list):
        deduplicated_list: list[Any] = []
        for item in value:
            deduplicated_item, next_ref = _deduplicate_report_value(
                item, seen, next_ref
            )
            deduplicated_list.append(deduplicated_item)
        return _deduplicate_report_container(deduplicated_list, seen, next_ref)

    return value, next_ref


def _deduplicate_report_container(
    value: dict[str, Any] | list[Any],
    seen: dict[str, str],
    next_ref: int,
) -> tuple[Any, int]:
    if not value:
        return value, next_ref

    fingerprint = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if fingerprint in seen:
        return {"$ref": seen[fingerprint], "$deduplicated_for_report": True}, next_ref

    ref = f"trace_value_{next_ref}"
    seen[fingerprint] = ref
    return value, next_ref + 1


def _conversation_events(
    spans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    displayed_message_count = 0

    for span in spans:
        attributes = span.get("attributes", {})
        if not isinstance(attributes, Mapping):
            continue
        if not _is_conversation_span(span, attributes):
            continue

        span_context = attributes.get("span_context", "")

        messages = attributes.get("all_messages", [])
        if isinstance(messages, list):
            for message in messages[displayed_message_count:]:
                if isinstance(message, Mapping):
                    msg = dict(message)
                    if span_context:
                        msg["span_context"] = span_context
                    events.append(msg)
            displayed_message_count = max(displayed_message_count, len(messages))

        response = attributes.get("response")
        response_message = _response_to_message(response)
        if response_message is not None:
            if span_context:
                response_message["span_context"] = span_context
            events.append(response_message)
            displayed_message_count += 1

    return events


def _is_conversation_span(
    span: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> bool:
    name = span.get("name")
    if name == "llm.complete":
        return True
    if not isinstance(name, str) or not name.startswith("llm.complete."):
        return False
    return attributes.get("llm_purpose") == "agent_turn"


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

    context = event.get("span_context", "")
    context_html = ""
    if context:
        context_html = (
            f'<span class="context-badge {escape(context)}">{escape(context)}</span>'
        )

    content_html = f"<pre>{escape(content_text)}</pre>" if content_text else ""
    return (
        f'<article class="message {escape(role_class)}">'
        f'<div class="role"><span>{escape(role)}{context_html}</span>'
        f"<span>{escape(meta)}</span></div>"
        f"{reasoning_html}{content_html}{tool_calls_html}</article>"
    )


def _render_span_summary(
    span: Mapping[str, Any],
) -> str:
    name = str(span.get("name") or "span")
    status = str(span.get("status") or "ok")
    duration = span.get("duration_ms")
    duration_text = f"{duration:.1f} ms" if isinstance(duration, int | float) else ""
    attributes = span.get("attributes", {})
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

    span_context = ""
    if isinstance(attributes, Mapping):
        span_context = attributes.get("span_context", "")

    context_class = ""
    if span_context:
        context_class = f" context-{escape(span_context)}"

    return (
        f'<div class="span{status_class}{context_class}">'
        f'<div class="span-title"><span>{escape(name)}</span>'
        f"<small>{escape(status)} {escape(duration_text)}</small></div>"
        f"<small>{escape(summary)}</small>{error_html}{details_html}</div>"
    )


def _span_attribute_summary(attributes: Any) -> str:
    if not isinstance(attributes, Mapping):
        return ""
    parts: list[str] = []
    llm_purpose = attributes.get("llm_purpose")
    if llm_purpose:
        parts.append(f"purpose={llm_purpose}")
    agent_mode = attributes.get("agent_mode")
    if agent_mode:
        parts.append(f"mode={agent_mode}")
    iteration = attributes.get("iteration")
    if isinstance(iteration, int):
        parts.append(f"iteration={iteration}")
    history_kind = attributes.get("history_kind")
    if history_kind:
        parts.append(f"history={history_kind}")
    tool_name = attributes.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")
    message_count = attributes.get("message_count")
    if message_count:
        parts.append(f"messages={message_count}")
    span_context = attributes.get("span_context")
    if span_context:
        parts.append(f"context={span_context}")
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
