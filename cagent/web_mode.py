"""Interactive web mode: serve the conversation over HTTP."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cagent.llm import LLMMessage
from cagent.modes import (
    BashAdvisor,
    _dispatch_tool_call,
    _make_clients,
    _tool_calls_with_ids,
)
from cagent.system_prompts import implementation_system_prompt
from cagent.tools import IMPLEMENTATION_TOOLS


@dataclass
class StoredMessage:
    """An LLM message with a stable ID for UI manipulation."""

    id: str
    message: LLMMessage


@dataclass
class WebSession:
    """Shared mutable state between the agent loop and HTTP handlers."""

    messages: list[StoredMessage] = field(default_factory=list)
    paused: bool = False
    waiting_for_user: bool = False
    busy: bool = False
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

    def remove_message(self, message_id: str) -> bool:
        with self.cond:
            for i, stored in enumerate(self.messages):
                if stored.id == message_id:
                    del self.messages[i]
                    self.cond.notify_all()
                    return True
            return False

    def set_paused(self, paused: bool) -> None:
        with self.cond:
            self.paused = paused
            self.cond.notify_all()

    def snapshot_messages(self) -> list[LLMMessage]:
        with self.cond:
            return [s.message for s in self.messages]

    def wait_until_ready(self) -> None:
        """Block while paused, or while there is no work for the LLM."""

        with self.cond:
            self.waiting_for_user = True
            try:
                while self.paused or not self._has_pending_work_locked():
                    self.cond.wait()
            finally:
                self.waiting_for_user = False

    def _has_pending_work_locked(self) -> bool:
        """Work exists when the latest message is a user input or tool result.

        A trailing assistant message (with no further tool results to feed back)
        is the "final answer" state — we wait for the next user message.
        """

        if not self.messages:
            return False
        last = self.messages[-1].message
        return last.role in ("user", "tool")


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
    }


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>cagent web</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 1em auto; padding: 0 1em; }
.msg { border: 1px solid #ccc; border-radius: 6px; padding: 8px 12px; margin: 8px 0; }
.user { background: #eef6ff; }
.assistant { background: #f6f6f6; }
.tool { background: #f0fff0; font-family: monospace; font-size: 0.9em; }
.system { background: #fff8e0; }
.role { font-weight: bold; font-size: 0.85em; color: #555; margin-bottom: 4px; }
.reasoning { font-style: italic; color: #666; white-space: pre-wrap; border-left: 3px solid #ccc; padding-left: 8px; margin-bottom: 4px; }
.content { white-space: pre-wrap; }
.tc { background: #fffde0; padding: 4px 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; }
.del { float: right; color: #c00; background: none; border: none; cursor: pointer; font-size: 1em; }
.status { padding: 8px; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 1em; position: sticky; top: 0; }
button { padding: 6px 12px; margin-right: 6px; cursor: pointer; }
textarea { width: 100%; min-height: 60px; font-family: inherit; padding: 6px; box-sizing: border-box; }
</style>
</head>
<body>
<div class="status">
  <span id="state">loading…</span>
  <button id="pauseBtn" onclick="togglePause()">Pause</button>
  <button onclick="refresh()">Refresh</button>
</div>
<div id="messages"></div>
<h3>Send message</h3>
<textarea id="input" placeholder="Type a message to add to the conversation..."></textarea>
<div style="margin-top:6px"><button onclick="sendMessage()">Send</button></div>
<script>
let paused = false;
async function refresh() {
  const r = await fetch('/api/state');
  const s = await r.json();
  paused = s.paused;
  document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
  let label;
  if (s.busy) label = 'running';
  else if (s.paused) label = 'paused';
  else if (s.waiting_for_user) label = 'waiting for input';
  else label = 'idle';
  document.getElementById('state').textContent = 'status: ' + label + ' | messages: ' + s.messages.length;
  const c = document.getElementById('messages');
  c.innerHTML = '';
  for (const m of s.messages) {
    const d = document.createElement('div');
    d.className = 'msg ' + m.role;
    let html = '<button class="del" onclick="removeMsg(\\''+m.id+'\\')">x</button>';
    html += '<div class="role">'+m.role+(m.tool_call_id?(' [tool_call_id='+escapeHtml(m.tool_call_id)+']'):'')+'</div>';
    if (m.reasoning) html += '<div class="reasoning">'+escapeHtml(m.reasoning)+'</div>';
    if (m.content) html += '<div class="content">'+escapeHtml(m.content)+'</div>';
    for (const tc of m.tool_calls) {
      html += '<div class="tc">→ '+escapeHtml(tc.name)+'('+escapeHtml(JSON.stringify(tc.arguments))+')</div>';
    }
    d.innerHTML = html;
    c.appendChild(d);
  }
}
function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function togglePause() {
  await fetch(paused ? '/api/resume' : '/api/pause', {method: 'POST'});
  refresh();
}
async function sendMessage() {
  const t = document.getElementById('input');
  const v = t.value;
  if (!v.trim()) return;
  await fetch('/api/message', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({content: v})});
  t.value = '';
  refresh();
}
async function removeMsg(id) {
  await fetch('/api/remove', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: id})});
  refresh();
}
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


def _make_handler(session: WebSession) -> type[BaseHTTPRequestHandler]:
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
                    payload = {
                        "paused": session.paused,
                        "busy": session.busy,
                        "waiting_for_user": session.waiting_for_user,
                        "messages": [
                            _message_to_dict(s) for s in session.messages
                        ],
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
    """Drop orphan tool messages whose preceding assistant tool_call was removed."""

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


def _run_web_loop(
    session: WebSession,
    *,
    fast_client: Any,
    smart_client: Any,
    bash_client: Any,
    system_prompt: str,
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
                return
            session.busy = True
        try:
            messages = _normalize_for_llm(session.snapshot_messages())
            try:
                response = fast_client.complete(
                    "",
                    system_prompt=system_prompt,
                    tools=IMPLEMENTATION_TOOLS,
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
                        session.cond.wait()
                content = _dispatch_tool_call(
                    tool_call,
                    bash_client=bash_client,
                    smart_client=smart_client,
                    fast_client=fast_client,
                    task_summary=task_summary,
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
                session.busy = False
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

    handler_cls = _make_handler(session)
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
            system_prompt=implementation_system_prompt(),
            task_summary=task_summary,
        )
    except KeyboardInterrupt:
        logging.info("Shutting down web mode")
    finally:
        server.shutdown()
        server.server_close()


__all__ = ["run_web_mode", "WebSession"]
