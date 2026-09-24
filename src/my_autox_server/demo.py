"""Loopback-only inspector for the AutoX phone agent.

Mirrors :mod:`jev_ultrafast.demo`: a tiny HTTP server on 127.0.0.1,
serialised via a lock, serving the inspector assets and the ``/api/*``
endpoints the UI calls.

The two big differences from jev-ultrafast's demo:

* Default port is ``8767`` (env override: ``AUTOX_DEMO_PORT``) so the phone
  agent and the browser agent can run side-by-side.
* The "scenario" dropdown lists phone tasks, not web tasks. The unit of
  work is "the device I point AutoX at", not "the URL I open in Chrome".
* A FakeAutoX backend is wired up by default so the inspector works
  locally without a phone; ``AGENT_USE_FAKE=0`` switches to a live
  :class:`AutoX` (which needs ``AUTOX_MCP_URL``).
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import Agent
from .autox import AutoX, FakeAutoX
from .questions import MAX_STEPS

ROOT = Path(__file__).parent
PORT = int(os.environ.get("AUTOX_DEMO_PORT", "8767"))
ORIGIN = f"http://127.0.0.1:{PORT}"
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
AGENT = None

SCENARIOS = {
    "search": (
        "Open the catalog search and type 'iPhone 15 case', "
        "then stop when results are showing."
    ),
    "settings": (
        "Open the Settings app, drill into the 'About phone' entry. "
        "Stop when the Android version is visible."
    ),
}


def load_environment():
    path = Path.cwd() / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


def make_device(label: str):
    """Pick :class:`FakeAutoX` or :class:`AutoX` based on ``AGENT_USE_FAKE``.

    Default is the real :class:`AutoX` so a fresh ``cp .env.example .env``
    is enough to drive a phone; switch to ``AGENT_USE_FAKE=1`` when you
    want to run the inspector without a device. ``CLIENT_INTEGRATION.md``
    documents the same flag.
    """
    use_fake = os.environ.get("AGENT_USE_FAKE", "0")
    if use_fake == "1":
        return FakeAutoX(label)
    return AutoX(label)


def response_state():
    state = (
        AGENT.snapshot()
        if AGENT
        else {"page": None, "status": "idle", "history": [], "decision": None, "elements": []}
    )
    return {
        **state,
        "text_model": os.environ.get("TEXT_MODEL", "deepseek-chat"),
        "max_steps": MAX_STEPS,
    }


def close_agent():
    global AGENT
    if AGENT:
        AGENT.close()
        AGENT = None


def command(name, body):
    global AGENT
    if name == "reset":
        scenario = body.get("scenario", "search")
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown demo scenario: {scenario}")
        goal = body.get("goal", "").strip()
        if not goal or len(goal) > 2000:
            raise ValueError("Enter 1–2,000 characters")
        close_agent()
        device = make_device(f"phone:{scenario}")
        AGENT = Agent(
            url=f"demo://{scenario}",
            goals=goal,
            screenshots=True,
            record_dir=Path.cwd() / "artifacts" / "frames" if body.get("record") else None,
            device=device,
        )
        AGENT.state["scenario"] = scenario
    else:
        if AGENT is None:
            raise ValueError("Start a demo first")
        AGENT.command(name, body)
    return response_state()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, mime="application/json"):
        content = content if isinstance(content, bytes) else content.encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{PORT}":
            return self.send(403, "Forbidden", "text/plain")
        path = urlparse(self.path).path
        if path == "/api/state":
            with LOCK:
                return self.send(200, json.dumps(response_state()))
        files = {
            "/": ("index.html", "text/html"),
            "/app.js": ("app.js", "text/javascript"),
            "/style.css": ("style.css", "text/css"),
        }
        if path not in files:
            return self.send(404, "Not found", "text/plain")
        name, mime = files[path]
        content = (ROOT / "static" / name).read_text().replace("__TOKEN__", TOKEN)
        self.send(200, content, mime + "; charset=utf-8")

    def do_POST(self):
        if (
            self.headers.get("Host") != f"127.0.0.1:{PORT}"
            or self.headers.get("X-Demo-Token") != TOKEN
            or self.headers.get("Origin") not in (None, ORIGIN)
        ):
            return self.send(403, json.dumps({"error": "Local demo requests only"}))
        if not LOCK.acquire(blocking=False):
            return self.send(409, json.dumps({"error": "A device step is already running"}))
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 8192:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            result = command(self.path.removeprefix("/api/"), body)
            self.send(200, json.dumps(result))
        except (ValueError, RuntimeError, TimeoutError) as error:
            self.send(400, json.dumps({"error": str(error)}))
        except Exception:
            self.send(500, json.dumps({"error": "Local demo failed; no automatic retry. Reset to recover."}))
        finally:
            LOCK.release()

    def log_message(self, *_args):
        pass


def main():
    load_environment()
    atexit.register(close_agent)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"my-autox-server: {ORIGIN}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
