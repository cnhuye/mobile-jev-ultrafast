"""Offline smoke tests. No MCP server, no paid model.

These tests exercise the device layer (FakeAutoX) and the agent's
predicate logic that drives ``tick`` / ``predict`` / ``act``. They do
*not* call TypeSafe or any text helper.
"""

from __future__ import annotations

import pytest

from mobile_jev_ultrafast import Agent, FakeAutoX
from mobile_jev_ultrafast.autox import _build_actions, _build_text


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


def test_agent_short_circuits_on_high_confidence_task_complete(monkeypatch):
    """``task_complete=finish`` above the threshold stops the loop.

    Mirrors the operational-goal case (e.g. 「上滑 1 下」): one action
    executes, then the model marks the step as terminal, and the
    agent loop must return ``status=done`` instead of feeding a
    second predict.
    """
    fake = FakeAutoX()

    def fake_choose(page, goal, history, *, task_plan=None, relevant_apps=None):
        # ``SCROLL_UP`` lives in ``controls`` (it's a synthetic action,
        # not a clickable element), so target is ``None`` and the
        # executor picks the action directly off ``action["id"]``.
        return {
            "choice": "scroll_up",
            "operation": "SCROLL_UP",
            "target": None,
            "confidence": 0.9,
            "probabilities": {"scroll_up": 1.0},
            "operation_probabilities": {"SCROLL_UP": 1.0},
            "target_probabilities": {},
            "target_confidence": None,
            "task_complete": "finish",
            "task_complete_confidence": 0.92,
            "task_complete_probabilities": {"continue": 0.08, "finish": 0.92},
            "raw_answers": {},
            "model": "fake",
            "usage": {},
            "latency_ms": 0,
            "request": {},
            "backend": "fake",
        }

    monkeypatch.setattr("mobile_jev_ultrafast.agent.choose", fake_choose)

    with Agent("demo://x", "上滑 1 下", device=fake) as agent:
        for _state in agent.run():
            pass

    # Exactly one decision reached ``done`` — no second predict fires
    # because the task_complete head short-circuits the loop.
    assert agent.state["status"] == "done"
    assert len(agent.state["decisions"]) == 1
    # The executed action records the task_complete head so downstream
    # consumers can see why the loop ended.
    last_action = agent.state["history"][-1]
    assert last_action["task_complete"] == "finish"
    assert last_action["task_complete_confidence"] == pytest.approx(0.92)


def test_agent_keeps_looping_when_task_complete_finish_is_low_confidence(monkeypatch):
    """Below-threshold ``task_complete=finish`` does NOT short-circuit.

    Guards against accidentally trusting low-confidence signals — a
    noisy Jev reply that says finish=0.4 must still let the loop run.
    """
    fake = FakeAutoX()
    calls = {"n": 0}

    def fake_choose(page, goal, history, *, task_plan=None, relevant_apps=None):
        calls["n"] += 1
        if calls["n"] >= 4:
            # Force an exit on the 4th call so the test doesn't loop forever.
            return {
                "choice": "DONE",
                "operation": "DONE",
                "target": None,
                "confidence": 1.0,
                "probabilities": {"DONE": 1.0},
                "operation_probabilities": {"DONE": 1.0},
                "target_probabilities": {},
                "target_confidence": None,
                "task_complete": "finish",
                "task_complete_confidence": 1.0,
                "task_complete_probabilities": {"continue": 0.0, "finish": 1.0},
                "raw_answers": {},
                "model": "fake",
                "usage": {},
                "latency_ms": 0,
                "request": {},
                "backend": "fake",
            }
        # Low confidence finish — should NOT stop the loop.
        return {
            "choice": "scroll_up",
            "operation": "SCROLL_UP",
            "target": None,
            "confidence": 0.9,
            "probabilities": {"scroll_up": 1.0},
            "operation_probabilities": {"SCROLL_UP": 1.0},
            "target_probabilities": {},
            "target_confidence": None,
            "task_complete": "finish",
            "task_complete_confidence": 0.4,  # below the 0.7 threshold
            "task_complete_probabilities": {"continue": 0.6, "finish": 0.4},
            "raw_answers": {},
            "model": "fake",
            "usage": {},
            "latency_ms": 0,
            "request": {},
            "backend": "fake",
        }

    monkeypatch.setattr("mobile_jev_ultrafast.agent.choose", fake_choose)

    with Agent("demo://x", "上滑 1 下", device=fake) as agent:
        for _state in agent.run():
            pass

    # The low-confidence finish must NOT short-circuit; the loop runs
    # until our forced DONE on the 4th call.
    assert calls["n"] >= 2, f"expected multiple predicts, got {calls['n']}"
    assert agent.state["status"] == "done"
