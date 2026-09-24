"""Offline tests for the MCP HTTP client.

Uses ``httpx.MockTransport`` so we never need a real phone. The fake
transport follows the wire format the AutoX.js server actually uses
per ``tools/list`` (direct ``result.tools`` array, no ``content[]``
wrapper) versus ``tools/call`` (result wrapped in ``content[]``).
"""

from __future__ import annotations

import json

import httpx
import pytest

from mobile_jev_ultrafast.mcp_client import MCPClient, _flatten_ocr


def _ui_tree_fixture() -> dict:
    """Compact UI tree as documented in ``MCP_USAGE.md`` §"get_ui_tree"."""
    return {
        "p": "com.android.settings",
        "activity": "Settings$NetworkDashboardActivity",
        "c": "FrameLayout",
        "b": [0, 0, 1080, 2400],
        "a": "",
        "children": [
            {
                "c": "Button",
                "id": "com.android.settings:id/search_button",
                "t": "Search",
                "b": [910, 100, 1050, 200],
                "a": "c",
                "children": [],
            },
            {
                "c": "EditText",
                "id": "com.android.settings:id/search_src_text",
                "t": "",
                "b": [80, 200, 1000, 320],
                "a": "c,f",
                "children": [],
            },
            {
                "c": "Switch",
                "t": "Notifications",
                "b": [80, 400, 1000, 520],
                "a": "c",
                "checked": False,
                "children": [],
            },
            {
                "c": "Spinner",
                "t": "All categories",
                "b": [80, 600, 1000, 720],
                "a": "c",
                "children": [],
            },
            {
                "c": "android.widget.LinearLayout",
                "t": "",
                "b": [0, 800, 1080, 1200],
                "a": "c",
                "children": [],
            },
        ],
    }


# ---------- response builders ----------------------------------------------

def _tools_list_response(*tool_names):
    """``tools/list`` returns tools directly under ``result.tools``."""
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": name} for name in tool_names]},
        },
    )


def _tools_list_response_empty():
    return _tools_list_response()


def _tool_call_response(value, is_error=False):
    """``tools/call`` returns its payload inside ``content[0].text``."""
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": json.dumps(value)}],
                "isError": is_error,
            },
        },
    )


def _make_mock_client(call_responses):
    """Build a client whose MCP responses come from a queue.

    ``tools/list`` is auto-replied with an empty list (the test can
    override by being more specific with the handler below) so most
    tests can ignore it.
    """
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        method = json.loads(request.content).get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if state["i"] < len(call_responses):
            resp = call_responses[state["i"]]
            state["i"] += 1
            return resp
        return _tool_call_response("ok")

    transport = httpx.MockTransport(handler)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    return MCPClient("http://mcp.test/mcp", token="t", initialize=False, client=http)


def _make_capturing_client(handler):
    """Low-level build that lets tests record every tools/call exactly."""
    transport = httpx.MockTransport(handler)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    return MCPClient("http://mcp.test/mcp", token="t", initialize=False, client=http)


# ---------- getUiTree --------------------------------------------------------

def test_get_ui_tree_parses_compact_format():
    client = _make_mock_client([_tool_call_response(_ui_tree_fixture())])
    out = client.get_ui_tree()
    assert out["p"] == "com.android.settings"
    assert out["children"][0]["t"] == "Search"


# ---------- screenshot / display_size ---------------------------------------

def test_screenshot_returns_base64():
    client = _make_mock_client([_tool_call_response({"data": "BASE64BYTES=="})])
    assert client.screenshot() == "BASE64BYTES=="


def test_display_size_falls_back_when_field_missing():
    client = _make_mock_client([])
    assert client.display_size() == (1080, 2400)


def test_display_size_reads_device_info():
    client = _make_mock_client(
        [_tool_call_response({"display": {"width": 1440, "height": 3200}})]
    )
    assert client.display_size() == (1440, 3200)


# ---------- camelCase tool names + error envelope ---------------------------

def test_set_text_prefers_native_tool():
    """When the server registers a ``set_text`` tool we call it directly."""
    captured = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response("set_text")
        if method == "tools/call":
            captured["name"] = body["params"]["name"]
            captured["args"] = body["params"]["arguments"]
        return _tool_call_response("ok")

    client = _make_capturing_client(handler)
    client.set_text("hello world")
    assert captured["name"] == "set_text"
    assert captured["args"] == {"text": "hello world"}


def test_set_text_falls_back_to_run_script():
    """Without a ``set_text`` tool, type via ``run_script`` with AutoX.js setText."""
    captured = {"name": None, "args": None}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if method == "tools/call":
            captured["name"] = body["params"]["name"]
            captured["args"] = body["params"]["arguments"]
        return _tool_call_response("ok")

    client = _make_capturing_client(handler)
    client.set_text("hello world")
    assert captured["name"] == "run_script"
    assert "setText(" in captured["args"]["script"]
    assert "hello world" in captured["args"]["script"]


def test_is_error_payload_is_raised():
    def handler(request: httpx.Request):
        method = json.loads(request.content).get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        return _tool_call_response("no permission", is_error=True)

    client = _make_capturing_client(handler)
    with pytest.raises(RuntimeError, match="no permission"):
        client.tap(100, 200)


def test_set_clipboard_prefers_set_clip():
    captured = {"name": None}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response("set_clip")
        if method == "tools/call":
            captured["name"] = body["params"]["name"]
        return _tool_call_response("ok")

    client = _make_capturing_client(handler)
    client.set_clipboard("payload")
    assert captured["name"] == "set_clip"


def test_set_clipboard_falls_back_to_run_script():
    captured = {"name": None, "args": None}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if method == "tools/call":
            captured["name"] = body["params"]["name"]
            captured["args"] = body["params"]["arguments"]
        return _tool_call_response("ok")

    client = _make_capturing_client(handler)
    client.set_clipboard("payload")
    assert captured["name"] == "run_script"
    assert "payload" in captured["args"]["script"]


# ---------- transport-level guarantees --------------------------------------

def test_initialize_round_trip_is_optional():
    """MCP server returning 200 to ``initialize`` must not break the client."""
    seen = {"init": False}

    def handler(request: httpx.Request):
        method = json.loads(request.content).get("method")
        if method == "initialize":
            seen["init"] = True
        if method == "tools/list":
            return _tools_list_response_empty()
        return _tool_call_response("ok")

    transport = httpx.MockTransport(handler)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    MCPClient("http://mcp.test/mcp", token=None, initialize=True, client=http)
    assert seen["init"] is True


def test_authorisation_token_is_sent_as_x_token():
    captured = {}

    def handler(request: httpx.Request):
        method = json.loads(request.content).get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if method == "tools/call":
            captured["token"] = request.headers.get("X-Token")
        return _tool_call_response("ok")

    transport = httpx.MockTransport(handler)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    client = MCPClient("http://mcp.test/mcp", token="secret", initialize=False, client=http)
    client.tap(1, 2)
    assert captured["token"] == "secret"


# ---------- ocr ----------------------------------------------------------

_OCR_TREE = [
    {
        "level": 0,
        "text": "微信",
        "bounds": {"top": 80, "left": 200, "bottom": 160, "right": 880},
        "confidence": 0.97,
        "language": "zh",
        "children": [
            {
                "level": 1,
                "text": "通讯录",
                "bounds": {"top": 2208, "left": 320, "bottom": 2288, "right": 488},
                "confidence": 0.92,
                "language": "zh",
                "children": [],
            }
        ],
    },
    {
        "level": 0,
        "text": "Discover",
        "bounds": {"top": 2208, "left": 600, "bottom": 2288, "right": 760},
        "confidence": 0.88,
        "language": "en",
        "children": [],
    },
]


def test_flatten_ocr_unwraps_nested_tree():
    flat = list(_flatten_ocr(_OCR_TREE))
    texts = [entry["text"] for entry in flat]
    assert texts == ["微信", "通讯录", "Discover"]
    # Bounds propagate.
    assert flat[0]["bounds"]["left"] == 200
    assert flat[1]["bounds"]["top"] == 2208
    assert flat[0]["confidence"] == pytest.approx(0.97)


def test_flatten_ocr_unwraps_result_envelope():
    wrapped = {"result": _OCR_TREE}
    flat = list(_flatten_ocr(wrapped))
    assert [entry["text"] for entry in flat] == ["微信", "通讯录", "Discover"]


def test_flatten_ocr_handles_string_payload():
    flat = list(_flatten_ocr(json.dumps(_OCR_TREE)))
    assert [entry["text"] for entry in flat] == ["微信", "通讯录", "Discover"]


def test_ocr_calls_mcp_tool_with_screenshot_source():
    captured = {"name": None, "args": None}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if method == "tools/call":
            captured["name"] = body["params"]["name"]
            captured["args"] = body["params"]["arguments"]
        return _tool_call_response(_OCR_TREE)

    client = _make_capturing_client(handler)
    out = client.ocr()
    assert captured["name"] == "ocr"
    assert captured["args"] == {"source": "screenshot", "language": "zh"}
    assert [entry["text"] for entry in out] == ["微信", "通讯录", "Discover"]


def test_ocr_uses_path_when_supplied():
    captured = {"args": None}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        if method == "tools/call":
            captured["args"] = body["params"]["arguments"]
        return _tool_call_response([])

    client = _make_capturing_client(handler)
    client.ocr(path="/sdcard/t.jpeg", language="en")
    assert captured["args"] == {"path": "/sdcard/t.jpeg", "language": "en"}


def test_ocr_returns_empty_list_when_tool_missing():
    """When the server doesn't expose ``ocr``, we degrade to ``[]``."""

    def handler(request: httpx.Request):
        method = json.loads(request.content).get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        return _tool_call_response("Tool ocr not registered", is_error=True)

    client = _make_capturing_client(handler)
    assert client.ocr() == []


# ---------- probe_ringer_mode ---------------------------------------------

def test_probe_ringer_mode_maps_audio_manager_codes():
    """``run_script`` is fire-and-forget, so the value rides an exception."""

    for code, expected in (("0", "silent"), ("1", "vibrate"), ("2", "ring")):
        def handler(request: httpx.Request):
            body = json.loads(request.content)
            method = body.get("method")
            if method == "tools/list":
                return _tools_list_response_empty()
            name = body["params"]["name"]
            if name == "run_script":
                return _tool_call_response({"jobId": 7})
            if name == "job_status":
                return _tool_call_response(
                    {"jobId": 7, "status": "FAILED", "message": f"Error: RINGER:{code} ..."}
                )
            return _tool_call_response("ok")

        client = _make_capturing_client(handler)
        assert client.probe_ringer_mode() == expected


def test_probe_ringer_mode_returns_none_when_script_ok():
    """A SUCCESS job carries no value, so the probe must give up cleanly."""

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        method = body.get("method")
        if method == "tools/list":
            return _tools_list_response_empty()
        name = body["params"]["name"]
        if name == "run_script":
            return _tool_call_response({"jobId": 7})
        if name == "job_status":
            return _tool_call_response({"jobId": 7, "status": "SUCCESS", "message": "completed"})
        return _tool_call_response("ok")

    client = _make_capturing_client(handler)
    assert client.probe_ringer_mode() is None


def test_probe_ringer_mode_survives_transport_failure():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if body.get("method") == "tools/list":
            return _tools_list_response_empty()
        return _tool_call_response("nope", is_error=True)

    client = _make_capturing_client(handler)
    assert client.probe_ringer_mode() is None
