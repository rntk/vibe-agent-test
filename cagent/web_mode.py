"""Interactive web mode: serve the conversation over HTTP."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

from cagent.advisor import AdvisorObserver
from cagent.compaction import is_tombstone, tombstone_text
from cagent.llm import LLMMessage, ToolDefinition
from cagent.modes import (
    BashAdvisor,
    _dispatch_tool_call,
    _make_clients,
    _tool_calls_with_ids,
)
from cagent.system_prompts import implementation_system_prompt
from cagent.tools import IMPLEMENTATION_TOOLS

_UNSET = object()

LoopStatus = Literal[
    "idle",
    "paused",
    "waiting_for_user",
    "done_waiting_for_user",
    "waiting_for_lm",
    "running_tool",
    "stopping",
]


@dataclass(frozen=True)
class ContextSize:
    """Estimated context size for the next LLM request."""

    characters: int
    estimated_tokens: int


@dataclass
class StoredMessage:
    """An LLM message with a stable ID for UI manipulation."""

    id: str
    message: LLMMessage
    kind: str = "message"  # "message" | "advisor"
    advisor_group: str | None = None  # e.g. "Edit precheck", "Bash precheck"


@dataclass
class WebSession:
    """Shared mutable state between the agent loop and HTTP handlers."""

    messages: list[StoredMessage] = field(default_factory=list)
    paused: bool = False
    status: LoopStatus = "idle"
    active_detail: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.cond = threading.Condition(self.lock)

    def add_message(self, message: LLMMessage) -> StoredMessage:
        with self.cond:
            stored = StoredMessage(id=uuid.uuid4().hex, message=message)
            self.messages.append(stored)
            self.cond.notify_all()
            return stored

    def add_user_message(self, content: str) -> StoredMessage:
        return self.add_message(LLMMessage(role="user", content=content))

    def add_advisor_message(
        self, message: LLMMessage, *, group: str
    ) -> StoredMessage:
        with self.cond:
            stored = StoredMessage(
                id=uuid.uuid4().hex,
                message=message,
                kind="advisor",
                advisor_group=group,
            )
            self.messages.append(stored)
            self.cond.notify_all()
            return stored

    def remove_message(self, message_id: str) -> bool:
        with self.cond:
            for i, stored in enumerate(self.messages):
                if stored.id == message_id:
                    if stored.kind == "advisor":
                        del self.messages[i]
                        self.cond.notify_all()
                        return True
                    msg = stored.message
                    if msg.role == "tool":
                        # Tombstone tool results instead of deleting them so
                        # the corresponding assistant tool_call still has a
                        # matching tool message in the list.
                        tool_name = self._lookup_tool_name(msg.tool_call_id)
                        new_msg = LLMMessage(
                            role="tool",
                            content=tombstone_text(tool_name, "deleted by user"),
                            tool_call_id=msg.tool_call_id,
                        )
                        self.messages[i] = StoredMessage(
                            id=stored.id, message=new_msg
                        )
                        if msg.tool_call_id:
                            self._strip_signature_if_any_tombstoned(
                                msg.tool_call_id
                            )
                    else:
                        # User and assistant messages are removed entirely.
                        del self.messages[i]
                    self.cond.notify_all()
                    return True
            return False

    def _lookup_tool_name(self, tool_call_id: str | None) -> str:
        """Return the tool name for *tool_call_id*, or ``"unknown"``."""

        if not tool_call_id:
            return "unknown"
        for stored in self.messages:
            msg = stored.message
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_call_id:
                        return tc.name
        return "unknown"

    def _strip_signature_if_any_tombstoned(self, tool_call_id: str) -> None:
        """Strip *thought_signature* from the assistant that issued
        *tool_call_id* when any of its tool results are tombstoned.

        The signature is cryptographically bound to the complete tool-call
        chain; once any result is tombstoned the chain is broken and the
        signature becomes stale.  This matches the compaction behaviour
        (see compact_history in compaction.py).

        *reasoning* is preserved because some providers (e.g. DeepSeek)
        require it to be passed back in every subsequent turn.
        """

        # Find the assistant message that owns this tool_call_id.
        assistant_idx = -1
        for idx, stored in enumerate(self.messages):
            msg = stored.message
            if msg.role == "assistant" and msg.tool_calls:
                ids = {tc.id for tc in msg.tool_calls if tc.id}
                if tool_call_id in ids:
                    assistant_idx = idx
                    break

        if assistant_idx < 0:
            return

        stored = self.messages[assistant_idx]
        msg = stored.message

        # Nothing to strip if there is no signature.
        if msg.thought_signature is None:
            return

        self.messages[assistant_idx] = StoredMessage(
            id=stored.id,
            message=LLMMessage(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
                reasoning=msg.reasoning,
                thought_signature=None,
            ),
        )

    def set_paused(self, paused: bool) -> None:
        with self.cond:
            self.paused = paused
            if paused and not self.busy:
                self.set_status_locked("paused", detail="Paused by user")
            elif self.is_waiting_for_user:
                self.set_status_locked(
                    self._waiting_status_locked(),
                    detail=self._waiting_detail_locked(),
                )
            elif not self.busy:
                self.set_status_locked("idle", detail=None)
            self.cond.notify_all()

    def set_status(
        self,
        status: LoopStatus,
        *,
        detail: str | None | object = _UNSET,
    ) -> None:
        with self.cond:
            self.set_status_locked(status, detail=detail)
            self.cond.notify_all()

    def set_status_locked(
        self,
        status: LoopStatus,
        *,
        detail: str | None | object = _UNSET,
    ) -> None:
        """Update status while the caller already holds ``self.cond``."""

        self.status = status
        if detail is not _UNSET:
            self.active_detail = detail

    @property
    def busy(self) -> bool:
        return self.status in ("waiting_for_lm", "running_tool")

    @property
    def is_waiting_for_user(self) -> bool:
        return self.status in ("waiting_for_user", "done_waiting_for_user")

    def snapshot_messages(self) -> list[LLMMessage]:
        with self.cond:
            return [s.message for s in self.messages]

    def wait_until_ready(self) -> None:
        """Block while paused, or while there is no work for the LLM."""

        with self.cond:
            if self.paused:
                self.set_status_locked("paused", detail="Paused by user")
            else:
                self.set_status_locked(
                    self._waiting_status_locked(),
                    detail=self._waiting_detail_locked(),
                )
            self.cond.notify_all()
            try:
                while self.paused or not self._has_pending_work_locked():
                    self.cond.wait()
                    if self.paused:
                        self.set_status_locked("paused", detail="Paused by user")
                    else:
                        self.set_status_locked(
                            self._waiting_status_locked(),
                            detail=self._waiting_detail_locked(),
                        )
            finally:
                self.set_status_locked("idle", detail=None)
                self.cond.notify_all()

    def _has_pending_work_locked(self) -> bool:
        """Work exists when the latest message is a user input or tool result.

        A trailing assistant message (with no further tool results to feed back)
        is the "final answer" state — we wait for the next user message.
        """

        if not self.messages:
            return False
        last = self.messages[-1].message
        return last.role in ("user", "tool")

    def _is_done_locked(self) -> bool:
        if not self.messages:
            return False
        last = self.messages[-1].message
        return last.role == "assistant" and not last.tool_calls

    def _waiting_status_locked(self) -> LoopStatus:
        if self._is_done_locked():
            return "done_waiting_for_user"
        return "waiting_for_user"

    def _waiting_detail_locked(self) -> str:
        if self._is_done_locked():
            return "Task done; waiting for user input"
        if not self.messages:
            return "Waiting for initial user input"
        return "Waiting for user input"


def _message_to_dict(stored: StoredMessage) -> dict[str, Any]:
    msg = stored.message
    tool_calls = [
        {
            "id": tc.id,
            "name": tc.name,
            "arguments": dict(tc.arguments) if tc.arguments else {},
        }
        for tc in (msg.tool_calls or ())
    ]
    return {
        "id": stored.id,
        "role": msg.role,
        "content": msg.content,
        "reasoning": msg.reasoning,
        "tool_calls": tool_calls,
        "tool_call_id": msg.tool_call_id,
        "kind": stored.kind,
        "advisor_group": stored.advisor_group,
    }


def _value_size(value: Any) -> int:
    """Estimate serialized size without allocating one large request string."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, int | float):
        return len(str(value))
    if isinstance(value, dict):
        return sum(len(str(key)) + _value_size(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return sum(_value_size(item) for item in value)
    return len(str(value))


def _context_size(
    messages: Sequence[LLMMessage],
    *,
    system_prompt: str,
    tools: Sequence[ToolDefinition],
) -> ContextSize:
    """Estimate the full request context web mode will send to the LLM."""

    characters = len(system_prompt)
    for tool in tools:
        characters += len(tool.name)
        characters += len(tool.description)
        characters += _value_size(tool.parameters)
        characters += _value_size(tool.strict)

    for message in messages:
        characters += len(message.role)
        characters += _value_size(message.content)
        characters += _value_size(message.reasoning)
        characters += _value_size(message.tool_call_id)
        for tool_call in message.tool_calls:
            characters += _value_size(tool_call.id)
            characters += len(tool_call.name)
            characters += _value_size(tool_call.arguments)

    return ContextSize(
        characters=characters,
        estimated_tokens=(characters + 3) // 4 if characters else 0,
    )


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>cagent web</title>
<style>
body {
  font-family: -apple-system, sans-serif;
  max-width: 900px;
  margin: 1em auto;
  padding: 0 1em;
}
.msg { border: 1px solid #ccc; border-radius: 6px; padding: 8px 12px; margin: 8px 0; }
.user { background: #eef6ff; }
.assistant { background: #f6f6f6; }
.tool { background: #f0fff0; font-family: monospace; font-size: 0.9em; }
.system { background: #fff8e0; }
.advisor-wrapper {
  border-left: 3px solid #9b59b6;
  margin: 4px 0 4px 16px;
  padding-left: 8px;
}
.advisor-header {
  font-size: 0.78em;
  font-weight: bold;
  color: #7d3c98;
  margin-bottom: 2px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.advisor-msg { border: 1px solid #d2b4de; border-radius: 4px; padding: 6px 10px; margin: 3px 0; }
.advisor-user { background: #f5eef8; }
.advisor-assistant { background: #f0e6f6; }
.advisor-tool { background: #e8daef; font-family: monospace; font-size: 0.85em; }
.role { font-weight: bold; font-size: 0.85em; color: #555; margin-bottom: 4px; }
.reasoning {
  font-style: italic;
  color: #666;
  white-space: pre-wrap;
  border-left: 3px solid #ccc;
  padding-left: 8px;
  margin-bottom: 4px;
}
.content { white-space: pre-wrap; }
.tc {
  background: #fffde0;
  padding: 4px 8px;
  margin: 4px 0;
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.85em;
  white-space: pre-wrap;
}
.del {
  float: right;
  color: #c00;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1em;
}
.status {
  padding: 8px;
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 1em;
  position: sticky;
  top: 0;
  z-index: 1;
}
.status-line {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 6px;
}
.metric { color: #555; font-size: 0.9em; }
button { padding: 6px 12px; margin-right: 6px; cursor: pointer; }
textarea {
  width: 100%;
  min-height: 60px;
  font-family: inherit;
  padding: 6px;
  box-sizing: border-box;
}
</style>
</head>
<body>
<div class="status">
  <div class="status-line">
    <span id="state">loading…</span>
    <span id="context" class="metric"></span>
    <span id="messagesMetric" class="metric"></span>
  </div>
  <div>
    <button id="pauseBtn" onclick="togglePause()">Pause</button>
    <button onclick="refresh()">Refresh</button>
  </div>
</div>
<div id="messages"></div>
<h3>Send message</h3>
<textarea
  id="input"
  placeholder="Type a message to add to the conversation..."
></textarea>
<div style="margin-top:6px"><button onclick="sendMessage()">Send</button></div>
<script>
let paused = false;
async function refresh() {
  const r = await fetch('/api/state');
  const s = await r.json();
  paused = s.paused;
  document.getElementById('pauseBtn').textContent =
    paused ? 'Resume' : 'Pause';
  const labels = {
    idle: 'idle',
    paused: 'paused',
    waiting_for_user: 'waiting for user input',
    done_waiting_for_user: 'done, waiting for user input',
    waiting_for_lm: 'waiting for LM response',
    running_tool: 'running tool',
    stopping: 'stopping'
  };
  const label = labels[s.status] || s.status || 'idle';
  const contextSize = s.context_size || {};
  const estimatedTokens = contextSize.estimated_tokens || 0;
  const characters = contextSize.characters || 0;
  document.getElementById('state').textContent =
    'status: ' + label + (s.active_detail ? ' - ' + s.active_detail : '');
  document.getElementById('context').textContent =
    'context: ~' + estimatedTokens.toLocaleString() +
    ' tokens / ' + characters.toLocaleString() + ' chars';
  document.getElementById('messagesMetric').textContent =
    'messages: ' + s.messages.length;
  const c = document.getElementById('messages');
  c.innerHTML = '';
  // Group consecutive advisor messages under a shared wrapper.
  let advisorWrapper = null;
  let advisorGroup = null;
  for (const m of s.messages) {
    if (m.kind === 'advisor') {
      if (advisorWrapper === null || m.advisor_group !== advisorGroup) {
        advisorWrapper = document.createElement('div');
        advisorWrapper.className = 'advisor-wrapper';
        const hdr = document.createElement('div');
        hdr.className = 'advisor-header';
        hdr.textContent = '🤖 ' + (m.advisor_group || 'Advisor');
        advisorWrapper.appendChild(hdr);
        c.appendChild(advisorWrapper);
        advisorGroup = m.advisor_group;
      }
      const d = document.createElement('div');
      d.className = 'advisor-msg advisor-' + m.role;
      let html = '<button class="del" onclick="removeMsg(\\''+m.id+'\\')">x</button>';
      html += '<div class="role">' + m.role +
        (m.tool_call_id ? (' [' + escapeHtml(m.tool_call_id) + ']') : '') +
        '</div>';
      if (m.reasoning) {
        html += '<div class="reasoning">' + escapeHtml(m.reasoning) + '</div>';
      }
      if (m.content) {
        html += '<div class="content">' + escapeHtml(m.content) + '</div>';
      }
      for (const tc of m.tool_calls) {
        html += '<div class="tc">-> ' + escapeHtml(tc.name) + '(' +
          escapeHtml(JSON.stringify(tc.arguments)) + ')</div>';
      }
      d.innerHTML = html;
      advisorWrapper.appendChild(d);
    } else {
      advisorWrapper = null;
      advisorGroup = null;
      const d = document.createElement('div');
      d.className = 'msg ' + m.role;
      let html =
        '<button class="del" onclick="removeMsg(\\''+m.id+'\\')">x</button>';
      html += '<div class="role">' + m.role +
        (m.tool_call_id ? (' [tool_call_id=' + escapeHtml(m.tool_call_id) + ']') : '') +
        '</div>';
      if (m.reasoning) {
        html += '<div class="reasoning">' + escapeHtml(m.reasoning) + '</div>';
      }
      if (m.content) {
        html += '<div class="content">' + escapeHtml(m.content) + '</div>';
      }
      for (const tc of m.tool_calls) {
        html += '<div class="tc">-> ' + escapeHtml(tc.name) + '(' +
          escapeHtml(JSON.stringify(tc.arguments)) + ')</div>';
      }
      d.innerHTML = html;
      c.appendChild(d);
    }
  }
}
function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[c]));
}
async function togglePause() {
  await fetch(paused ? '/api/resume' : '/api/pause', {method: 'POST'});
  refresh();
}
async function sendMessage() {
  const t = document.getElementById('input');
  const v = t.value;
  if (!v.trim()) return;
  await fetch('/api/message', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({content: v})
  });
  t.value = '';
  refresh();
}
async function removeMsg(id) {
  await fetch('/api/remove', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id: id})
  });
  refresh();
}
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


def _make_handler(
    session: WebSession,
    *,
    system_prompt: str,
    tools: Sequence[ToolDefinition],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                with session.cond:
                    stored_messages = list(session.messages)
                    paused = session.paused
                    busy = session.busy
                    waiting_for_user = session.is_waiting_for_user
                    status = session.status
                    active_detail = session.active_detail
                messages = _normalize_stored_for_llm(stored_messages)
                context_size = _context_size(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                )
                payload = {
                    "paused": paused,
                    "busy": busy,
                    "waiting_for_user": waiting_for_user,
                    "status": status,
                    "active_detail": active_detail,
                    "context_size": {
                        "characters": context_size.characters,
                        "estimated_tokens": context_size.estimated_tokens,
                    },
                    "messages": [_message_to_dict(s) for s in stored_messages],
                }
                self._json(200, payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/message":
                data = self._read_json()
                content = (data.get("content") or "").strip()
                if not content:
                    self._json(400, {"error": "empty content"})
                    return
                stored = session.add_user_message(content)
                self._json(200, {"id": stored.id})
                return
            if self.path == "/api/remove":
                data = self._read_json()
                mid = data.get("id") or ""
                ok = session.remove_message(mid)
                self._json(200 if ok else 404, {"removed": ok})
                return
            if self.path == "/api/pause":
                session.set_paused(True)
                self._json(200, {"paused": True})
                return
            if self.path == "/api/resume":
                session.set_paused(False)
                self._json(200, {"paused": False})
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def _normalize_for_llm(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Drop orphan tool messages and advisor-only messages."""

    valid_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    valid_ids.add(tc.id)

    out: list[LLMMessage] = []
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id not in valid_ids:
            continue
        out.append(msg)
    return out


def _normalize_stored_for_llm(stored_messages: list[StoredMessage]) -> list[LLMMessage]:
    """Filter advisor messages, then apply LLM normalization."""

    main_messages = [s.message for s in stored_messages if s.kind == "message"]
    return _normalize_for_llm(main_messages)


def _run_web_loop(
    session: WebSession,
    *,
    fast_client: Any,
    smart_client: Any,
    bash_client: Any,
    system_prompt: str,
    tools: Sequence[ToolDefinition],
    task_summary: str,
) -> None:
    iteration = 0
    while True:
        session.wait_until_ready()

        with session.cond:
            last_user = next(
                (s for s in reversed(session.messages) if s.message.role == "user"),
                None,
            )
            if last_user and (last_user.message.content or "").strip() == "/exit":
                logging.info("Received /exit — stopping web loop")
                print("Received /exit — stopping web loop")
                session.set_status_locked("stopping", detail="Received /exit")
                session.cond.notify_all()
                return
        try:
            with session.cond:
                stored_snapshot = list(session.messages)
            messages = _normalize_stored_for_llm(stored_snapshot)
            try:
                session.set_status(
                    "waiting_for_lm",
                    detail=f"Agent turn {iteration + 1}",
                )
                response = fast_client.complete(
                    "",
                    system_prompt=system_prompt,
                    tools=tools,
                    messages=messages,
                    trace_name="llm.complete.web.agent_turn",
                    trace_attributes={
                        "llm_purpose": "agent_turn",
                        "agent_mode": "web",
                        "iteration": iteration,
                    },
                )
            except Exception as exc:
                logging.exception("LLM call failed: %s", exc)
                session.add_message(
                    LLMMessage(role="user", content=f"[LLM error: {exc}]")
                )
                continue

            tool_calls = _tool_calls_with_ids(response.tool_calls or (), iteration)
            session.add_message(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                    reasoning=response.reasoning,
                    thought_signature=response.thought_signature,
                )
            )
            iteration += 1

            if not tool_calls:
                # Final assistant turn — wait for next user message.
                continue

            for tool_call in tool_calls:
                # Check pause between tool calls so the user can intervene.
                with session.cond:
                    while session.paused:
                        session.set_status_locked(
                            "paused",
                            detail="Paused between tool calls",
                        )
                        session.cond.notify_all()
                        session.cond.wait()
                session.set_status(
                    "running_tool",
                    detail=tool_call.name,
                )

                def _on_advisor(message: LLMMessage, group: str) -> None:
                    session.add_advisor_message(message, group=group)
                    with session.cond:
                        session.set_status_locked(
                            "running_tool",
                            detail=f"Advisor: {group}",
                        )
                        session.cond.notify_all()

                content = _dispatch_tool_call(
                    tool_call,
                    bash_client=bash_client,
                    smart_client=smart_client,
                    fast_client=fast_client,
                    task_summary=task_summary,
                    on_advisor=_on_advisor,
                )
                session.add_message(
                    LLMMessage(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call.id,
                    )
                )
        finally:
            with session.cond:
                if session.status in ("waiting_for_lm", "running_tool"):
                    session.set_status_locked("idle", detail=None)
                session.cond.notify_all()


def run_web_mode(
    file_path: str | None,
    *,
    bash_advisor: BashAdvisor = "off",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the interactive web mode server and run the agent loop."""

    fast_client, smart_client, bash_client = _make_clients(bash_advisor)

    session = WebSession()
    if file_path:
        initial = Path(file_path).read_text(encoding="utf-8")
        session.add_user_message(initial)
        task_summary = initial
    else:
        task_summary = ""

    system_prompt = implementation_system_prompt()
    handler_cls = _make_handler(
        session,
        system_prompt=system_prompt,
        tools=IMPLEMENTATION_TOOLS,
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    server_thread = threading.Thread(
        target=server.serve_forever, name="cagent-web", daemon=True
    )
    server_thread.start()
    logging.info("Web UI listening on http://%s:%d", host, port)
    print(f"Web UI: http://{host}:{port}")

    try:
        _run_web_loop(
            session,
            fast_client=fast_client,
            smart_client=smart_client,
            bash_client=bash_client,
            system_prompt=system_prompt,
            tools=IMPLEMENTATION_TOOLS,
            task_summary=task_summary,
        )
    except KeyboardInterrupt:
        logging.info("Shutting down web mode")
    finally:
        server.shutdown()
        server.server_close()


__all__ = ["run_web_mode", "WebSession"]
