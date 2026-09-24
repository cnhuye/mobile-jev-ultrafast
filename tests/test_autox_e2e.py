"""End-to-end tests with a mock MCP server.

Wires :class:`AutoX` up to an ``httpx.MockTransport`` that pretends to be
the AutoX.js MCP server. The point is to make sure
``AutoX.observe`` / ``AutoX.act`` produce the same shape that the
inspector and the agent loop expect, given the UI tree documented in
``../docs/MCP_USAGE.md``.
"""

from __future__ import annotations

import json

import httpx

from my_autox_server.autox import (
    AutoX,
    _classify,
    _interaction_flags,
    _node_id,
    _node_kind,
)
from my_autox_server.mcp_client import MCPClient

_COMPACT_TREE = {
    "p": "com.android.settings",
    "activity": "Settings",
    "c": "FrameLayout",
    "b": [0, 0, 1080, 2400],
    "a": "",
    "children": [
        {
            "c": "android.widget.Button",
            "t": "Search settings",
            "b": [80, 100, 480, 220],
            "a": "clickable=true",
            "children": [],
        },
        {
            "c": "EditText",
            "t": "",
            "b": [80, 260, 1000, 380],
            "a": "c,f",
            "children": [],
        },
        {
            "c": "Switch",
            "t": "Notifications",
            "b": [80, 420, 1000, 540],
            "a": "c",
            "checked": True,
            "children": [],
        },
        {
            "c": "Spinner",
            "t": "All categories",
            "b": [80, 580, 1000, 700],
            "a": "c",
            "children": [],
        },
        {
            "c": "TextView",
            "t": "Header",
            "b": [0, 0, 1080, 80],
            "children": [],
        },
    ],
}


def _ok_response(value, is_error=False) -> httpx.Response:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps(value)}],
            "isError": is_error,
        },
    }
    return httpx.Response(200, json=body)


def _tools_list_empty() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []},
        },
    )


class _CapturingTransport(httpx.MockTransport):
    """Mock transport that records every ``tools/call`` and replies from
    a queue. ``tools/list`` is auto-replied with an empty tool list so
    tests don't have to bookkeep it."""

    def __init__(self, call_responses):
        self.calls = []
        self.queue = list(call_responses)
        super().__init__(self._dispatch)

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get("method")
        params = body.get("params") or {}
        if method == "tools/list":
            return _tools_list_empty()
        if method == "tools/call":
            self.calls.append((params.get("name"), params.get("arguments") or {}))
            if self.queue:
                resp = self.queue.pop(0)
                if callable(resp):
                    return resp(self.calls[-1])
                return resp
            return _ok_response("ok")
        return _ok_response("ok")


def _make_autox_with(call_responses):
    """Build an AutoX whose MCP round-trips feed the given queue.

    Pre-pends an empty ``deviceInfo`` response so ``display_size()``
    doesn't burn the first ``getUiTree`` slot when it asks for display
    info at construction time.
    """
    # The client's constructor calls ``deviceInfo`` once via
    # ``display_size()``. Empty payload → fallback to 1080x2400. The
    # rest of ``call_responses`` is returned in order.
    transport = _CapturingTransport([_ok_response({})] + list(call_responses))
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    client = MCPClient("http://mcp.test/mcp", initialize=False, client=http)
    return AutoX("device:test", mcp=client), transport


# ---------- _interaction_flags / _classify ----------------------------------

def test_interaction_flags_accept_verbose_legacy_format():
    assert "c" in _interaction_flags({"a": "clickable=true"})


def test_interaction_flags_accept_compact_symbols():
    flags = _interaction_flags({"a": "c, f"})
    assert "c" in flags and "f" in flags


def test_interaction_flags_handle_comma_separated_string():
    flags = _interaction_flags({"a": "c,f"})
    assert "c" in flags and "f" in flags


def test_classify_accepts_full_and_short_class_names():
    assert _classify({"c": "android.widget.Button", "a": ""}) == "button"
    assert _classify({"c": "Button", "a": ""}) == "button"
    assert _classify({"c": "android.widget.EditText", "a": ""}) == "textbox"
    assert _classify({"c": "Switch", "a": ""}) == "switch"


def test_classify_promotes_clickable_layout_to_button():
    assert _classify({"c": "android.widget.LinearLayout", "a": "c"}) == "button"


def test_node_kind_picks_fill_for_textboxes():
    node = {"c": "EditText", "a": "c,f", "b": [0, 0, 100, 100]}
    assert _node_kind(node, _classify(node)) == "fill"


def test_node_id_is_deterministic():
    a = {"b": [10, 20, 30, 40], "c": "Button", "t": "Go", "d": "", "id": ""}
    b = {"b": [10, 20, 30, 40], "c": "Button", "t": "Go", "d": "", "id": ""}
    assert _node_id(a) == _node_id(b)
    other = dict(a)
    other["t"] = "Back"
    assert _node_id(a) != _node_id(other)


# ---------- AutoX.observe against the documented UI tree -------------------

def test_autox_observe_yields_indexed_action_table():
    auto, _tx = _make_autox_with([_ok_response(_COMPACT_TREE)])
    page = auto.observe()
    # Each role maps to a specific action kind: button→click, textbox→fill,
    # switch→click, combobox→select. Look at all kinds together.
    role_kinds = {(a["role"], a["kind"]) for a in page["actions"] if a["id"].startswith("e")}
    assert ("button", "click") in role_kinds
    assert ("textbox", "fill") in role_kinds
    assert ("switch", "click") in role_kinds
    assert ("combobox", "select") in role_kinds
    synthetic = {
        a["id"]
        for a in page["actions"]
        if a["id"] in {"scroll_down", "scroll_up", "wait"}
    }
    assert synthetic == {"scroll_down", "scroll_up", "wait"}
    real_ids = [a["id"] for a in page["actions"] if a["id"].startswith("e")]
    assert real_ids == sorted(real_ids)
    auto.close()


def test_autox_observe_extracts_package_and_activity():
    auto, _tx = _make_autox_with([_ok_response(_COMPACT_TREE)])
    page = auto.observe()
    assert page["package"] == "com.android.settings"
    assert page["activity"] == "Settings"
    assert page["url"] == "com.android.settings/Settings"
    auto.close()


def test_autox_act_tap_calls_mcp_tap():
    # Queue responses in order:
    #   ``[0]`` display_size's ``deviceInfo`` (consumed by AutoX.__init__).
    #   ``[1]`` first ``getUiTree`` for ``observe()``.
    #   ``[2]`` second ``getUiTree`` for the ``fresh()`` check inside ``act``.
    #   tap then rides the empty fallback.
    responses = [_ok_response(_COMPACT_TREE), _ok_response(_COMPACT_TREE)]
    auto, tx = _make_autox_with(responses)
    page = auto.observe()
    click_action = next(
        a for a in page["actions"] if a["kind"] == "click" and a["role"] == "button"
    )
    auto.act(click_action, page)
    names = [name for name, _args in tx.calls]
    # display_size only fires ``device_info`` once at construction time;
    # tools/list is filtered out and never reaches the captured list.
    assert names[:1] == ["device_info"]
    assert names[1:3] == ["get_ui_tree", "get_ui_tree"]
    assert names[-1] == "tap"
    expected_x = click_action["rect"]["x"] + click_action["rect"]["w"] // 2
    expected_y = click_action["rect"]["y"] + click_action["rect"]["h"] // 2
    assert tx.calls[-1][1]["x"] == expected_x
    assert tx.calls[-1][1]["y"] == expected_y
    auto.close()


def test_autox_act_fill_calls_set_text():
    """End-to-end: tapping a textbox then typing falls back to run_script.

    The MCP server advertised by the phone today doesn't ship a
    ``set_text`` tool, so ``AutoX.act(fill, text=...)`` falls back to
    ``run_script`` with an inline AutoX.js snippet.
    """
    responses = [
        _ok_response(_COMPACT_TREE),
        _ok_response(_COMPACT_TREE),
    ]
    auto, tx = _make_autox_with(responses)
    page = auto.observe()
    fill_action = next(a for a in page["actions"] if a["kind"] == "fill")
    auto.act(fill_action, page, text="hello")
    tool_names = [name for name, _args in tx.calls]
    # device_info, get_ui_tree, get_ui_tree, tap (focus), run_script (fallback)
    assert tool_names == [
        "device_info",
        "get_ui_tree",
        "get_ui_tree",
        "tap",
        "run_script",
    ]
    assert "setText(" in tx.calls[-1][1]["script"]
    assert "hello" in tx.calls[-1][1]["script"]
    auto.close()
