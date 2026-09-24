"""HTTP client for the AutoX.js MCP server.

The MCP wire format is JSON-RPC 2.0 over HTTP. ``tools/call`` requests
carry the tool name plus its arguments; results are wrapped in a
``content`` array of text/image blobs. ``getUiTree`` returns a compact
JSON tree; ``tap`` / ``swipe`` / ``setText`` etc. take coordinates or
text.

Tool names mirror the AutoX.js build described in
``../docs/MCP_USAGE.md`` and ``../docs/CLIENT_INTEGRATION.md``: they are
**camelCase** (``getUiTree``, ``setText``, ``setClip``, ``screenshot``)
and ``getUiTree`` returns ``{c, id, t, d, b, a, checked, children,
...}`` per the compact UI tree spec.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_S = 15.0
DEFAULT_RETRIES = 3

# Headers are required by the Streamable HTTP transport the MCP server
# implements. ``Accept`` lists both content types so the server can pick
# JSON (its current behaviour) without us having to renegotiate.
_ACCEPT = "application/json, text/event-stream"


def _flatten_ocr(node):
    """Yield every ``{text, bounds, confidence}`` entry from an OCR tree.

    The MCP ``ocr`` tool returns ``runtime.gmlkit.ocr()`` output verbatim:
    a nested array of up to three levels, where each leaf has a
    ``text``, ``confidence`` and ``bounds`` object. Some builds also wrap
    the array in ``{"result": [...]}; we accept both shapes.
    """
    if node is None:
        return
    if isinstance(node, str):
        try:
            node = json.loads(node)
        except (ValueError, TypeError):
            return
    if isinstance(node, dict):
        for key in ("result", "data", "ocr"):
            if key in node:
                yield from _flatten_ocr(node[key])
                return
        text = node.get("text")
        if isinstance(text, str) and text:
            yield {
                "text": text,
                "confidence": float(node.get("confidence", 0.0) or 0.0),
                "bounds": node.get("bounds") or {},
            }
        for child in node.get("children") or []:
            yield from _flatten_ocr(child)
        return
    if isinstance(node, list):
        for entry in node:
            yield from _flatten_ocr(entry)
        return
    # Primitive fallback (numbers, bools, etc.) — nothing to yield.


class MCPClient:
    """Synchronous MCP client. httpx handles connection pooling & http2."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        timeout: float = DEFAULT_TOOL_TIMEOUT_S,
        client: httpx.Client | None = None,
        initialize: bool = True,
    ):
        # Default URL scheme points at the AutoX.js MCP endpoint
        # (e.g. ``http://192.168.1.100:27190/mcp``). Older installs may
        # not have the ``/mcp`` suffix; ``MCPClient.from_env`` lets the
        # user supply the exact path they verified with curl.
        self.url = url.rstrip("/")
        self.token = token or ""
        self._owns_client = client is None
        self._client = client or httpx.Client(http2=True, timeout=timeout)
        if initialize:
            try:
                self._initialize()
            except Exception as exc:  # noqa: BLE001 — bootstrap only
                # Some AutoX builds don't require an ``initialize`` round
                # trip. If the server is happy with bare ``tools/call``,
                # we silently fall back.
                log.debug("initialize failed; continuing without handshake: %s", exc)
        self._tools = self._discover_tools()

    @classmethod
    def from_env(cls):
        url = os.environ.get("AUTOX_MCP_URL")
        if not url:
            raise RuntimeError(
                "AUTOX_MCP_URL is not set. Set it to the phone's MCP endpoint, e.g. "
                "AUTOX_MCP_URL=http://192.168.1.100:27190/mcp"
            )
        token = os.environ.get("AUTOX_MCP_TOKEN") or None
        return cls(url, token=token)

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "Accept": _ACCEPT,
        }
        if self.token:
            h["X-Token"] = self.token
        return h

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Invoke one MCP tool. Returns the parsed JSON-RPC ``result`` payload.

        Raises :class:`RuntimeError` on transport or HTTP failure so the
        agent loop can react (no silent retries on user input).
        """
        body = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        last_err: Exception | None = None
        for attempt in range(DEFAULT_RETRIES):
            try:
                resp = self._client.post(
                    self.url, json=body, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                last_err = exc
                time.sleep(0.4 * (2**attempt))
                continue
            if resp.status_code in {429, 503, 529} and attempt < DEFAULT_RETRIES - 1:
                time.sleep(0.4 * (2**attempt))
                continue
            if resp.is_error:
                raise RuntimeError(
                    f"MCP server returned HTTP {resp.status_code} for {name}: {resp.text[:300]}"
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"MCP server returned non-JSON body for {name}: {resp.text[:300]}"
                ) from exc
            if "error" in data:
                err = data["error"]
                raise RuntimeError(
                    f"MCP tool {name} failed: {err.get('message')} (code={err.get('code')})"
                )
            result = data.get("result") or {}
            if result.get("isError"):
                content = result.get("content") or []
                msg = content[0].get("text", "") if content else ""
                raise RuntimeError(f"MCP tool {name} returned an error: {msg}")
            return result
        raise RuntimeError(
            f"MCP transport failed after {DEFAULT_RETRIES} attempts: {last_err}"
        )

    def _initialize(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "my-autox-server", "version": "0.1.0"},
            },
        }
        try:
            resp = self._client.post(self.url, json=body, headers=self._headers())
        except httpx.HTTPError:
            return
        if resp.status_code >= 400:
            return

    # ------------------------------------------------------------------
    # Tool discovery (best-effort)
    # ------------------------------------------------------------------

    def _discover_tools(self) -> dict[str, dict]:
        try:
            resp = self._client.post(
                self.url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers=self._headers(),
            )
            if resp.is_error:
                return {}
            tools = (resp.json().get("result") or {}).get("tools") or []
            return {t["name"]: t for t in tools if isinstance(t, dict) and "name" in t}
        except Exception:
            return {}

    @property
    def tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def close(self):
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Tool wrappers. Names mirror the canonical AutoX.js MCP build
    # described in ``../docs/MCP_USAGE.md``.
    # ------------------------------------------------------------------

    def _extract(self, result: dict) -> Any:
        """MCP returns ``content: [{type:"text", text:"<json|string>"}]``."""
        content = result.get("content") if isinstance(result, dict) else None
        if not content:
            return {}
        first = content[0] if isinstance(content, list) else content
        if not isinstance(first, dict):
            return {}
        if first.get("type") == "text":
            text = first.get("text") or ""
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return text
        if first.get("type") in {"image", "jpeg", "png"}:
            return first.get("data") or first.get("base64")
        return first

    # ---- perception ---------------------------------------------------

    def get_ui_tree(self) -> dict:
        """Fetch the compact UI tree via the ``get_ui_tree`` MCP tool.

        Note: earlier docs called this ``getUiTree`` (camelCase); the
        AutoX.js build on the phone registers it as ``get_ui_tree``
        (snake_case). We use the name ``tools/list`` advertises.
        """
        result = self.call_tool("get_ui_tree", {})
        data = self._extract(result)
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except ValueError:
                pass
        return {"text": data if isinstance(data, str) else "", "children": []}

    def screenshot(self, as_base64: bool = True) -> str | None:
        """Return a base64-encoded JPEG/PNG, or ``None`` if unavailable.

        The MCP ``screenshot`` tool accepts ``asBase64`` to choose between
        a temporary file path and base64 bytes. We always request base64
        because the inspector expects an inline data URI.

        Errors propagate: a ``SecurityException`` from the phone (e.g.
        missing screen capture permission) is raised so callers can
        surface the real cause. Returning ``None`` is reserved for a
        ``null``/empty successful payload, which is rare in practice.
        """
        result = self.call_tool("screenshot", {"asBase64": as_base64})
        data = self._extract(result)
        if isinstance(data, dict):
            return data.get("data") or data.get("base64")
        if isinstance(data, str):
            return data or None
        return None

    def get_recent_screenshot(self, as_base64: bool = True) -> str | None:
        try:
            result = self.call_tool("get_recent_screenshot", {"asBase64": as_base64})
        except RuntimeError:
            return self.screenshot(as_base64=as_base64)
        data = self._extract(result)
        if isinstance(data, dict):
            return data.get("data") or data.get("base64")
        return data if isinstance(data, str) else None

    def get_foreground_app(self) -> str:
        try:
            result = self.call_tool("get_foreground_app", {})
        except RuntimeError:
            return ""
        data = self._extract(result)
        if isinstance(data, dict):
            return data.get("packageName") or data.get("package") or ""
        return data if isinstance(data, str) else ""

    def get_current_activity(self) -> str:
        try:
            result = self.call_tool("get_current_activity", {})
        except RuntimeError:
            return ""
        data = self._extract(result)
        if isinstance(data, dict):
            return data.get("activity") or data.get("name") or ""
        return data if isinstance(data, str) else ""

    def ocr(self, *, source: str = "screenshot", path: str | None = None,
            language: str = "zh") -> list[dict]:
        """Run on-device OCR via Google ML Kit.

        Returns a flat list of ``{text, bounds, confidence}`` entries.
        The AutoX.js server returns the same nested shape that
        ``runtime.gmlkit.ocr()`` produces; we flatten it so callers can
        iterate one text element at a time.

        Either ``source="screenshot"`` (uses the last screenshot taken
        by the device) or an absolute ``path`` to an image file must be
        supplied. The default language is ``zh`` because the reference
        device is a HUAWEI LYA-AL00 with a Chinese locale; pass
        ``language="en"`` for English text.
        """
        args: dict = {"language": language}
        if path:
            args["path"] = path
        else:
            args["source"] = source
        try:
            result = self.call_tool("ocr", args)
        except RuntimeError:
            return []
        flat = self._extract(result)
        return list(_flatten_ocr(flat))

    # --- device state probes ------------------------------------------

    # AudioManager.RINGER_MODE_* -> a stable string. Used by
    # `examples/toggle_silent_mode.py` to verify the toggle independently
    # of the model's own DONE answer.
    _RINGER_MODES = {"0": "silent", "1": "vibrate", "2": "ring"}

    def probe_ringer_mode(self) -> str | None:
        """Return the phone's current ringer mode: silent / vibrate / ring.

        ``run_script`` on the AutoX.js side is fire-and-forget — it hands
        back a job id, never a value. So we run a one-line script that
        *throws* an exception whose message carries the AudioManager
        code, then read it back off ``job_status``. Ugly, but it is the
        only synchronous value channel the MCP server exposes today.

        Returns ``None`` when the probe fails for any reason, so callers
        can degrade instead of crashing.
        """
        script = (
            '"auto";'
            "var am = context.getSystemService(android.content.Context.AUDIO_SERVICE);"
            'throw new Error("RINGER:" + am.getRingerMode());'
        )
        try:
            result = self.run_script(script, name="probe_ringer", timeout_millis=3000)
        except RuntimeError:
            return None
        payload = self._extract(result)
        job_id = payload.get("jobId") if isinstance(payload, dict) else None
        if job_id is None:
            return None
        for _ in range(25):
            time.sleep(0.12)
            try:
                status = self._extract(self.call_tool("job_status", {"jobId": job_id}))
            except RuntimeError:
                return None
            if not isinstance(status, dict):
                continue
            if status.get("status") in {"SUCCESS", "FAILED"}:
                message = status.get("message") or ""
                match = re.search(r"RINGER:(\d+)", message)
                if match:
                    return self._RINGER_MODES.get(match.group(1))
                return None
        return None

    def device_info(self) -> dict:
        try:
            result = self.call_tool("device_info", {})
        except RuntimeError:
            return {}
        data = self._extract(result)
        return data if isinstance(data, dict) else {}

    # ---- input -------------------------------------------------------

    def tap(self, x: int, y: int) -> dict:
        return self.call_tool("tap", {"x": int(x), "y": int(y)})

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> dict:
        return self.call_tool(
            "swipe",
            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "duration": int(duration)},
        )

    def set_text(self, text: str) -> dict:
        """Type ``text`` into the currently focused EditText.

        Falls back to a ``run_script`` invocation that calls AutoX.js'
        built-in ``setText()`` (which writes through the Accessibility
        service) when the phone's MCP build doesn't ship a direct
        ``setText`` tool — which the v7 build on the phone does not.
        """
        if "set_text" in self._tools:
            return self.call_tool("set_text", {"text": text})
        script = (
            "(()=>{"
            "try{setText(" + json.dumps(text) + ");return 'ok';}"
            "catch(e){return String(e);}"
            "})()"
        )
        result = self.run_script(script, name="autox_setText", timeout_millis=5000)
        payload = self._extract(result) if isinstance(result, dict) else ""
        if isinstance(payload, str) and payload != "ok":
            raise RuntimeError(f"AutoX setText fallback failed: {payload}")
        return {"ok": True, "fallback": True}

    def set_clipboard(self, text: str) -> dict:
        """Set the system clipboard.

        Prefers a dedicated ``set_clip`` MCP tool; falls back to a
        ``run_script`` that writes the clipboard via AutoX.js' global
        ``setClip`` Java helper when the MCP build doesn't register it.
        """
        for name in ("set_clip", "set_clipboard"):
            if name in self._tools:
                return self.call_tool(name, {"text": text})
        script = (
            "(()=>{"
            "const c=runtime.getProperty('android.content.ClipboardManager');"
            "const ctx=context.getApplicationContext();"
            "if(typeof setClip==='function'){setClip(" + json.dumps(text) + ");return 'ok';}"
            "return 'no setClip';"
            "})()"
        )
        result = self.run_script(script, name="autox_setClip", timeout_millis=5000)
        payload = self._extract(result) if isinstance(result, dict) else ""
        if isinstance(payload, str) and payload != "ok":
            raise RuntimeError(f"AutoX setClip fallback failed: {payload}")
        return {"ok": True, "fallback": True}

    def find_element(
        self,
        *,
        text: str | None = None,
        id: str | None = None,
        desc: str | None = None,
        class_name: str | None = None,
        timeout: int = 0,
    ) -> dict:
        """Resolve a single node without scanning the UI tree client-side."""
        criteria = {}
        if text is not None:
            criteria["text"] = text
        if id is not None:
            criteria["id"] = id
        if desc is not None:
            criteria["desc"] = desc
        if class_name is not None:
            criteria["className"] = class_name
        try:
            result = self.call_tool(
                "find_element",
                {"criteria": criteria, "timeout": int(timeout)},
            )
        except RuntimeError:
            return {}
        data = self._extract(result)
        return data if isinstance(data, dict) else {}

    def find_elements(self, *, text: str | None = None, limit: int = 10) -> list:
        try:
            result = self.call_tool(
                "find_elements", {"text": text or "", "limit": int(limit)}
            )
        except RuntimeError:
            return []
        data = self._extract(result)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("elements"), list):
            return data["elements"]
        return []

    # ---- app/script control -----------------------------------------

    def app_control(self, action: str, package: str) -> dict:
        return self.call_tool("app_control", {"action": action, "package": package})

    def run_script(self, script: str, *, name: str | None = None,
                   mode: str = "v7", timeout_millis: int = 20000) -> dict:
        args: dict = {"script": script, "mode": mode, "timeoutMillis": int(timeout_millis)}
        if name:
            args["name"] = name
        return self.call_tool("run_script", args)

    # ---- display size derived from ``device_info`` ------------------

    def display_size(self) -> tuple[int, int]:
        """Best-effort display size. Falls back to 1080x2400.

        There's no dedicated ``getDisplaySize`` tool — display info comes
        out of ``deviceInfo``. Different builds disagree on field names,
        so we sniff several shapes before falling back.
        """
        info = self.device_info()
        for key in ("display", "screen", "resolution"):
            d = info.get(key)
            if isinstance(d, dict):
                w, h = d.get("width") or d.get("w"), d.get("height") or d.get("h")
                if isinstance(w, int) and isinstance(h, int):
                    return int(w), int(h)
        w = info.get("displayWidth") or info.get("screenWidth")
        h = info.get("displayHeight") or info.get("screenHeight")
        if isinstance(w, int) and isinstance(h, int):
            return int(w), int(h)
        return (1080, 2400)
