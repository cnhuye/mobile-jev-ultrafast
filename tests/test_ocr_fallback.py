"""Tests for the OCR fallback used when ``get_ui_tree`` is too sparse.

WeChat (and similar apps that block UI Automator) returns a tree
containing only the root node with ``a: "d"``. The compact tree
flattens to *zero* actionable elements, so the policy has nothing to
choose from. With ``ocr_fallback=True`` the device layer supplements
the empty tree with text + bounds from ``mcp.ocr(source="screenshot")``
and the agent loop can keep making progress.
"""

from __future__ import annotations

import json

import httpx
import pytest

from my_autox_server.autox import (
    MIN_REAL_ACTIONS,
    AutoX,
    _build_ocr_actions,
    _is_sparse_ui,
    _ocr_rect,
)
from my_autox_server.mcp_client import MCPClient

# ---- helpers --------------------------------------------------------------

_OCR_PAYLOAD = [
    {
        "level": 0,
        "text": "微信",
        "bounds": {"top": 80, "left": 200, "bottom": 160, "right": 880},
        "confidence": 0.97,
        "language": "zh",
        "children": [],
    },
    {
        "level": 0,
        "text": "通讯录",
        "bounds": {"top": 2208, "left": 320, "bottom": 2288, "right": 488},
        "confidence": 0.92,
        "language": "zh",
        "children": [],
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


_EMPTY_TREE = {
    "p": "com.tencent.mm",
    "activity": "com.tencent.mm.ui.LauncherUI",
    "c": "FrameLayout",
    "b": [0, 0, 1080, 2400],
    "a": "d",
    "children": [],
}


def _ok(value, is_error=False) -> httpx.Response:
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


def _tools_list_empty() -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})


class _MockTransport(httpx.MockTransport):
    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []
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
                return self.queue.pop(0)
        return _ok("ok")


def _make_autox(ocr_entries, *, ocr_fallback=False, ui_tree=_EMPTY_TREE):
    """Build an AutoX that reports an empty UI tree + OCR data."""
    # Responses served in order:
    #   [0] device_info for display_size
    #   [1] get_ui_tree (the empty tree)
    #   [2] ocr (when fallback is enabled)
    queue = [
        _ok({}),  # device_info
        _ok(ui_tree),  # get_ui_tree
    ]
    if ocr_entries is not None:
        queue.append(_ok(ocr_entries))
    transport = _MockTransport(queue)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    client = MCPClient("http://mcp.test/mcp", initialize=False, client=http)
    return AutoX("phone:test", mcp=client, ocr_fallback=ocr_fallback), transport


# ---- unit helpers ---------------------------------------------------------

def test_ocr_rect_converts_top_left_bottom_right():
    rect = _ocr_rect({"top": 100, "left": 200, "bottom": 160, "right": 880})
    assert rect == {"x": 200, "y": 100, "w": 680, "h": 60}


def test_ocr_rect_handles_missing_bounds():
    rect = _ocr_rect({})
    assert rect["w"] >= 1 and rect["h"] >= 1


def test_build_ocr_actions_emits_click_per_text():
    actions = _build_ocr_actions(_OCR_PAYLOAD)
    labels = [a["label"] for a in actions]
    assert labels == ["微信", "通讯录", "Discover"]
    for action in actions:
        assert action["kind"] == "click"
        assert action["id"].startswith("ocr")
        assert action["role"] == "text"
        assert action["ocr"] is True
    # Centre of the first entry: ((200+880)/2, (80+160)/2) = (540, 120).
    assert actions[0]["rect"]["x"] == 200
    assert actions[0]["rect"]["y"] == 80


def test_build_ocr_actions_skips_empty_entries():
    payload = [
        {"text": "", "bounds": {}},
        {"text": "OK", "bounds": {"top": 1, "left": 1, "bottom": 2, "right": 2}},
    ]
    actions = _build_ocr_actions(payload)
    assert [a["label"] for a in actions] == ["OK"]


def test_is_sparse_ui_flags_empty_state():
    state = {"actions": [{"id": "scroll_down"}, {"id": "scroll_up"}, {"id": "wait"}], "text": ""}
    assert _is_sparse_ui(state) is True


def test_is_sparse_ui_passes_when_text_rich():
    """A page with no actionable rows but lots of label text is still useful."""
    state = {"actions": [], "text": "About phone\nSettings\nNotifications"}
    assert _is_sparse_ui(state) is False


def test_is_sparse_ui_passes_when_actions_rich():
    state = {"actions": [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}], "text": ""}
    assert _is_sparse_ui(state) is False
    assert MIN_REAL_ACTIONS == 3


# ---- AutoX.observe with OCR fallback --------------------------------------

def test_observe_without_fallback_returns_almost_empty_actions_for_a11y_blocked_app():
    """An a11y-blocked tree still leaks one frame-layout action.

    The root FrameLayout with ``a:"d"`` (only the disabled flag) makes
    it through ``_build_actions`` as a single full-screen click
    target. That's *not* enough for the agent to make a real decision,
    which is exactly why the OCR fallback exists. Without OCR enabled
    the policy would have nothing useful to choose from.
    """
    auto, _ = _make_autox(ocr_entries=None, ocr_fallback=False)
    page = auto.observe()
    real = [a for a in page["actions"] if a["id"].startswith(("e", "ocr"))]
    # Just the frame-layout root; no actionable nodes with text.
    assert len(real) == 1
    assert real[0]["role"] == "framelayout"
    auto.close()


def test_observe_with_fallback_calls_ocr_when_tree_empty():
    auto, tx = _make_autox(ocr_entries=_OCR_PAYLOAD, ocr_fallback=True)
    page = auto.observe()
    real = [a for a in page["actions"] if a["id"].startswith(("e", "ocr"))]
    labels = [a["label"] for a in real]
    assert labels == ["微信", "通讯录", "Discover"]
    # The state carries the OCR provenance so callers can decide.
    assert page["fallback"] == "ocr"
    assert page["ocr_count"] == 3
    assert 0 < page["ocr_confidence_avg"] < 1
    auto.close()


def test_observe_with_fallback_skips_ocr_when_tree_is_rich():
    """OCR should only run when the tree itself is sparse."""
    rich_tree = {
        "p": "com.android.settings",
        "activity": "Settings",
        "c": "FrameLayout",
        "b": [0, 0, 1080, 2400],
        "a": "",
        "children": [
            {
                "c": "Button", "t": "Search", "b": [80, 100, 480, 220],
                "a": "clickable=true", "children": [],
            },
            {
                "c": "Button", "t": "About", "b": [80, 300, 480, 420],
                "a": "clickable=true", "children": [],
            },
            {
                "c": "Button", "t": "Help", "b": [80, 500, 480, 620],
                "a": "clickable=true", "children": [],
            },
        ],
    }
    queue = [_ok({}), _ok(rich_tree)]
    transport = _MockTransport(queue)
    http = httpx.Client(http2=False, timeout=5, transport=transport)
    client = MCPClient("http://mcp.test/mcp", initialize=False, client=http)
    auto = AutoX("phone:test", mcp=client, ocr_fallback=True)
    page = auto.observe()
    assert page.get("fallback") != "ocr"
    real = [a for a in page["actions"] if a["id"].startswith("e")]
    assert [a["label"] for a in real] == ["Search", "About", "Help"]
    auto.close()


def test_observe_with_fallback_uses_existing_certain_positions_for_ocr_actions():
    auto, _ = _make_autox(ocr_entries=_OCR_PAYLOAD, ocr_fallback=True)
    page = auto.observe()
    contacts = next(a for a in page["actions"] if a["label"] == "通讯录")
    cx = contacts["rect"]["x"] + contacts["rect"]["w"] // 2
    cy = contacts["rect"]["y"] + contacts["rect"]["h"] // 2
    # The original OCR bounds were (320,2208)-(488,2288), so the centre
    # is (404, 2248). That's the same coordinate the existing
    # ``wechat_contacts.py`` script taps by hand.
    assert (cx, cy) == (404, 2248)
    auto.close()


def test_observe_with_fallback_keeps_package_and_activity():
    auto, _ = _make_autox(ocr_entries=_OCR_PAYLOAD, ocr_fallback=True)
    page = auto.observe()
    assert page["package"] == "com.tencent.mm"
    assert page["activity"] == "com.tencent.mm.ui.LauncherUI"
    auto.close()


def test_observe_with_fallback_handles_empty_ocr_result():
    auto, _ = _make_autox(ocr_entries=[], ocr_fallback=True)
    page = auto.observe()
    assert page["fallback"] == "ocr"
    real = [a for a in page["actions"] if a["id"].startswith(("e", "ocr"))]
    assert real == []
    auto.close()


# ---------------------------------------------------------------------------
# Pytest discovery
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_env_overrides(monkeypatch):
    # The tests use ``httpx.MockTransport`` and never touch ``TYPESAFE_API_KEY``
    # etc., so we don't set env vars here.
    pass