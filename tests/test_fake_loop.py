"""Offline smoke tests. No MCP server, no paid model.

These tests exercise the device layer (FakeAutoX) and the agent's
predicate logic that drives ``tick`` / ``predict`` / ``act``. They do
*not* call TypeSafe or any text helper.
"""

from __future__ import annotations

import pytest

from my_autox_server import Agent, FakeAutoX
from my_autox_server.autox import _build_actions, _build_text


def _mock_node(role="button", label="Search", value="", bounds=(60, 200, 1020, 280)):
    return {
        "c": f"android.widget.{role.capitalize()}",
        "b": list(bounds),
        "t": label if role not in {"text"} else value,
        "d": "",
        "a": "clickable=true" if role != "textbox" else "clickable=true, focusable=true",
    }


def test_build_actions_classifies_widgets():
    tree = {
        "c": "android.widget.FrameLayout",
        "b": [0, 0, 1080, 2400],
        "a": "",
        "children": [
            _mock_node("button", "Search"),
            _mock_node("textbox", "Query", value=""),
            _mock_node("switch", "Notifications", value="true"),
        ],
    }
    actions = _build_actions(tree)
    kinds = [a["kind"] for a in actions if a["kind"] in {"click", "fill"}]
    roles = [a["role"] for a in actions if a["kind"] in {"click", "fill"}]
    assert "click" in kinds
    assert "fill" in kinds
    assert "button" in roles
    assert "textbox" in roles
    assert "switch" in roles
    ids = {a["id"] for a in actions}
    assert {"scroll_down", "scroll_up", "wait"} <= ids


def test_build_text_truncates():
    tree = {"c": "View", "b": [0, 0, 100, 100], "a": "", "children": [
        {"c": "TextView", "b": [0, 0, 100, 100], "t": "x" * 8000, "a": ""},
    ]}
    text = _build_text(tree)
    assert len(text) <= 6000


def test_fake_auto_observe_is_stable():
    fake = FakeAutoX()
    p1 = fake.observe()
    p2 = fake.observe()
    # Marker & fingerprint should be identical for the same mocked screen.
    assert p1["marker"] == p2["marker"]
    assert p1["fingerprint"] == p2["fingerprint"]
    assert p1["actions"], "expected some actions on the home screen"


def test_fake_auto_act_routes_transitions():
    fake = FakeAutoX()
    # Find the "Open Settings" button by label and act.
    state = fake.observe()
    settings_action = next(a for a in state["actions"] if a["label"] == "Open Settings")
    fake.act(settings_action, state)
    after = fake.observe()
    assert "Settings" in after["url"]


def test_agent_snapshot_includes_element_table():
    fake = FakeAutoX()
    with Agent("demo://x", "Some goal", device=fake) as agent:
        snap = agent.snapshot()
        assert snap["elements"], "element table should be populated"
        assert snap["status"] == "ready"
        # The agent never ran a model; snapshot's decision is None.
        assert snap["decision"] is None


def test_agent_command_blocks_when_stopped():
    fake = FakeAutoX()
    agent = Agent("demo://x", "Some goal", device=fake)
    try:
        # Mark finished by hand and ensure commands respect it.
        agent.state["status"] = "done"
        with pytest.raises(ValueError):
            agent.command("predict")
    finally:
        agent.close()
