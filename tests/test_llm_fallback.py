"""Tests for the LLM deadlock-breaker + post-run summary plumbing.

These tests exercise the agent-level wiring without ever calling the
auxiliary LLM — we monkey-patch :mod:`mobile_jev_ultrafast.llm` with a
fake so the loop runs deterministically.
"""

from __future__ import annotations

import os
from unittest import mock

# Use the scripted backend so the decision step never tries to reach
# the TypeSafe API in tests.
os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
# Drop a fake TypeSafe key so even an accidental ``_choose_typesafe``
# call would have something to attach to the request.
os.environ.setdefault("TYPESAFE_API_KEY", "test-fake-key")

from mobile_jev_ultrafast import Agent, FakeAutoX
from mobile_jev_ultrafast.autox import _build_actions


def _fake_llm():
    """Replace every LLM helper with a deterministic canned response."""
    return mock.patch.multiple(
        "mobile_jev_ultrafast.llm",
        plan_task=mock.Mock(
            return_value={
                "plan_text": "1. LAUNCH settings\n2. tap 声音和振动",
                "relevant_apps": ["设置"],
                "model": "fake",
            }
        ),
        decide_next_action=mock.Mock(
            return_value={
                "kind": "press_back",
                "text": "执行 PRESS_BACK 返回上一级",
                "model": "fake",
            }
        ),
        summarize_result=mock.Mock(
            return_value={"summary": "已成功切换铃声模式", "model": "fake"}
        ),
    )


def test_agent_runs_with_llm_plan_enabled():
    fake = FakeAutoX()
    with _fake_llm():
        with Agent(
            "demo://x",
            "Open Settings and tap About",
            device=fake,
            llm_plan=True,
            llm_fallback=True,
            llm_summary=True,
        ) as agent:
            assert agent.state["task_plan"].startswith("1. LAUNCH settings")
            assert "设置" in agent.state["plan_relevant_apps"]
            for _ in agent.run():
                pass
    # ``summarize_result`` is called by ``run`` on exit.
    assert agent.state.get("summary", {}).get("summary") == "已成功切换铃声模式"
    assert agent.state["status"] == "done"


def test_llm_fallback_does_not_trigger_when_actions_diverge():
    fake = FakeAutoX()
    fallback = mock.MagicMock(
        return_value={"kind": "press_back", "text": "执行 PRESS_BACK"}
    )
    plan = mock.MagicMock(
        return_value={"plan_text": "", "relevant_apps": [], "model": "fake"}
    )
    with mock.patch.multiple(
        "mobile_jev_ultrafast.llm",
        plan_task=plan,
        decide_next_action=fallback,
        summarize_result=mock.MagicMock(),
    ):
        with Agent(
            "demo://x",
            "Open Settings and tap About",
            device=fake,
            llm_plan=True,
            llm_fallback=True,
            llm_summary=False,
        ) as agent:
            for _ in agent.run():
                pass
    # The verified loop taps Settings → About; no deadlock, so the
    # fallback should never be called.
    assert fallback.call_count == 0
    assert plan.call_count == 1


def test_summary_off_keeps_state_unadorned():
    fake = FakeAutoX()
    with _fake_llm():
        with Agent(
            "demo://x",
            "Open Settings and tap About",
            device=fake,
            llm_summary=False,
        ) as agent:
            for _ in agent.run():
                pass
    assert "summary" not in agent.state


def test_deadlock_detector_flags_repeats():
    fake = FakeAutoX()
    # Inject a synthetic history so the detector sees three identical
    # action keys without any real model calls.
    with Agent("demo://x", "any", device=fake) as agent:
        agent.state["history"] = [
            {"step": i, "kind": "click", "choice": "e1", "action": "Same"}
            for i in range(1, 4)
        ]
        deadlock = agent._detect_deadlock()
    assert deadlock is not None
    assert deadlock["kind"] == "repeat"
    assert deadlock["action_key"] == "click:e1"


def test_deadlock_detector_ignores_scroll_only_window():
    fake = FakeAutoX()
    with Agent("demo://x", "any", device=fake) as agent:
        agent.state["history"] = [
            {"step": i, "kind": "scroll", "choice": "scroll_down", "action": "Scroll"}
            for i in range(1, 6)
        ]
        deadlock = agent._detect_deadlock()
    assert deadlock is not None
    assert deadlock["kind"] == "scroll_only"


def test_synthetic_actions_cover_new_kinds():
    """The element table now includes press/swipe/double_tap/wait_long."""
    actions = _build_actions(
        {"c": "FrameLayout", "b": [0, 0, 100, 100], "a": "", "children": []},
        launch_apps=None,
    )
    ids = {a["id"] for a in actions}
    for required in (
        "press_home",
        "press_back",
        "press_recents",
        "swipe_left",
        "swipe_right",
        "double_tap",
        "wait_long",
        "scroll_down",
        "scroll_up",
        "wait",
    ):
        assert required in ids, f"missing synthetic action {required}"
    launch_actions = [a for a in actions if a["kind"] == "launch"]
    assert launch_actions, "expected at least one launch_<pkg> action"


def test_llm_module_falls_back_when_no_api_key():
    """``plan_task`` / ``decide_next_action`` / ``summarize_result``
    must each degrade cleanly when ``TEXT_MODEL_API_KEY`` is unset so
    the agent loop never crashes."""
    from mobile_jev_ultrafast import llm

    fake = FakeAutoX()
    with mock.patch.dict("os.environ", {}, clear=True):
        # Remove the API key so the LLM path can't accidentally succeed.
        with mock.patch.object(llm, "_call_llm", side_effect=RuntimeError("no key")):
            plan = llm.plan_task("打开设置", fake)
            suggestion = llm.decide_next_action(
                goal="打开设置",
                page={"package": "com.x", "text": "", "actions": []},
                recent_actions=[],
                deadlock_note="stuck",
                enabled=True,
            )
            summary = llm.summarize_result(
                goal="打开设置",
                page={"text": "", "actions": []},
                logs=[],
                success=False,
                enabled=True,
            )
    assert plan["model"] == "scripted"
    assert suggestion["model"] == "scripted"
    assert "summary" in summary