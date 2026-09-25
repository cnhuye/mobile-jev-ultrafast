"""Tests for the tiered ``wait`` actions, especially ``WAIT_LONGER``.

Background
----------
After LAUNCH_APP, the target app almost always shows a splash / ad /
permission dialog before its real home screen renders. Without a
dedicated 3-second pause Jev either taps on the ad creative or picks
the post-ad screen as the "goal reached" baseline, breaking the rest
of the plan. The pre-task planner is now told to insert WAIT_LONGER
right after every LAUNCH_APP (see ``llm._PLAN_SYSTEM``), and the
deadlock-decide fallback suggests it when the most recent action was a
launch that hasn't had time to settle.
"""

from __future__ import annotations

import os

os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
os.environ.setdefault("TYPESAFE_API_KEY", "test-fake-key")

from mobile_jev_ultrafast.autox import (
    AutoX,
    _WAIT_DURATIONS,
    _build_actions,
    _synthetic_actions,
)
from mobile_jev_ultrafast.llm import (
    _PLAN_SYSTEM,
    _DECIDE_SYSTEM,
    _parse_decide_suggestion,
    _scripted_decide,
)


def test_wait_actions_table_includes_three_tiers():
    """All three wait tiers must exist with their labels and durations."""
    expected = {
        "wait": ("等待界面刷新", 0.1),
        "wait_long": ("等待 2 秒（动画/启动）", 2.0),
        "wait_longer": ("等待 5 秒（启动广告 / 启动页）", 5.0),
    }
    actions = {
        a["id"]: a for a in _synthetic_actions(launch_apps=None) if a["kind"] == "wait"
    }
    for action_id, (label, duration) in expected.items():
        assert action_id in actions, f"missing synthetic action {action_id}"
        assert actions[action_id]["label"] == label
        assert _WAIT_DURATIONS[action_id] == duration


def test_synthetic_actions_table_exposes_wait_longer():
    """The full action table (the one Jev sees) must include WAIT_LONGER
    so the model's operation criteria pick it up alongside WAIT/WAIT_LONG.
    """
    actions = _build_actions(
        {"c": "FrameLayout", "b": [0, 0, 100, 100], "a": "", "children": []},
        launch_apps=None,
    )
    ids = {a["id"] for a in actions}
    assert "wait" in ids
    assert "wait_long" in ids
    assert "wait_longer" in ids


def test_wait_longer_label_mentions_ad_or_splash():
    """The label is the only signal Jev has to decide between WAIT and
    WAIT_LONG/WAIT_LONGER. It must mention the use case explicitly so
    the model picks the right tier after LAUNCH_APP.
    """
    actions = {a["id"]: a for a in _synthetic_actions(launch_apps=None)}
    label = actions["wait_longer"]["label"]
    assert "5" in label and "秒" in label
    # Must reference the launch-ad / splash use case so Jev picks it
    # after LAUNCH_APP rather than the 0.1s / 2s tiers.
    assert ("广告" in label) or ("启动页" in label), label


def test_planner_prompt_requires_wait_longer_after_launch():
    """The planner system prompt must tell the LLM to insert WAIT_LONGER
    after every LAUNCH_APP. Without this hint the LLM would happily emit
    plans that go straight from LAUNCH_APP to CLICK, and Jev would land
    on the splash screen.
    """
    # The two key strings the planner must surface. Kept short so a
    # future copy-edit doesn't silently drop the rule.
    assert "WAIT_LONGER" in _PLAN_SYSTEM
    assert "LAUNCH_APP" in _PLAN_SYSTEM
    # The two must appear near each other so the rule reads as a single
    # instruction ("after LAUNCH_APP do WAIT_LONGER") rather than two
    # unrelated fragments scattered through the prompt.
    plan_idx = _PLAN_SYSTEM.find("WAIT_LONGER")
    launch_idx = _PLAN_SYSTEM.find("LAUNCH_APP", plan_idx - 200)
    assert 0 <= launch_idx < plan_idx, (
        "WAIT_LONGER must come after a LAUNCH_APP mention; otherwise the "
        "planner can't tell what action should be followed by the wait."
    )


def test_deadlock_prompt_mentions_wait_longer():
    """The deadlock-decide prompt must include WAIT_LONGER in its
    allowed-verbs list and show it in the examples, otherwise the LLM
    fallback can't tell the device layer to wait 3 s on a stuck
    splash screen.
    """
    assert "WAIT_LONGER" in _DECIDE_SYSTEM


def test_parse_decide_recognises_wait_longer_aliases():
    """The deadlock-decide parser must recognise both Latin
    (``WAIT_LONGER``) and Chinese (``等广告``, ``等启动``) aliases so a
    planner that replies in Chinese still maps cleanly onto the action.
    """
    for raw in (
        "执行 WAIT_LONGER 等广告结束",
        "等 5 秒",
        "等启动",
        "等广告",
    ):
        parsed = _parse_decide_suggestion(raw)
        assert parsed["kind"] == "wait_longer", parsed


def test_scripted_decide_suggests_wait_longer_after_launch():
    """If the most recent action was a ``launch`` (and no wait yet), the
    scripted fallback must return ``WAIT_LONGER`` instead of falling
    through to ``SCROLL_DOWN`` — otherwise the offline agent would
    scroll the splash screen instead of waiting for it to clear.
    """
    recent = [
        {"kind": "launch", "choice": "launch_com.taobao.idlefish", "action": "启动 闲鱼"},
        {"kind": "launch", "choice": "launch_com.taobao.idlefish", "action": "启动 闲鱼"},
    ]
    parsed = _scripted_decide(recent, {})
    assert parsed["kind"] == "wait_longer", parsed


def test_scripted_decide_does_not_force_wait_longer_unnecessarily():
    """The launch-then-wait shortcut must only fire when the most recent
    action really was a launch. After a CLICK deadlock we still want
    SCROLL_DOWN / PRESS_BACK — not a 3-second wait on a screen that
    has nothing to load.
    """
    recent = [
        {"kind": "click", "choice": "e1", "action": "Click something"},
        {"kind": "click", "choice": "e1", "action": "Click something"},
        {"kind": "click", "choice": "e1", "action": "Click something"},
    ]
    parsed = _scripted_decide(recent, {})
    assert parsed["kind"] != "wait_longer", parsed


def test_act_dispatches_wait_longer_to_five_seconds():
    """End-to-end: invoking the device with a ``wait_longer`` action
    must sleep for the duration declared in :data:`_WAIT_DURATIONS`.

    We patch ``time.sleep`` so the test stays fast and doesn't actually
    block for 3 s. The point is the dispatch wiring, not the wall-clock.
    """
    from mobile_jev_ultrafast import autox as autox_module
    from mobile_jev_ultrafast.autox import StalePage

    class _NoopMCP:
        def __init__(self):
            self.display = (1080, 2400)

        def display_size(self):
            return self.display

        def get_ui_tree(self):
            return {"c": "FrameLayout", "b": [0, 0, 1080, 2400], "a": "", "children": []}

        def screenshot(self, as_base64=False):
            return ""

        def installed_apps(self):
            return []

        # ``fresh()`` calls observe(), which calls get_ui_tree(). Keep
        # the tree stable so the freshness check passes.
        def installed_apps_cached(self):
            return []

    sleeps: list[float] = []
    real_sleep = autox_module.time.sleep

    def fake_sleep(seconds):
        sleeps.append(seconds)

    device = AutoX(mcp=_NoopMCP())
    try:
        # Stable page so fresh() returns True.
        page = device.observe(screenshot=False)
        # Find the WAIT_LONGER synthetic action on the observed page.
        action = next(
            a for a in page["actions"] if a["id"] == "wait_longer"
        )
        autox_module.time.sleep = fake_sleep
        try:
            result = device.act(action, page)
        finally:
            autox_module.time.sleep = real_sleep
        assert result == {"executed": "wait_longer"}
        assert sleeps == [5.0], sleeps
    finally:
        device.close()