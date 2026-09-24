"""Offline tests for the deterministic ``scripted`` decision backend.

These exist so that ``AGENT_DECISION_BACKEND=scripted`` can be relied
upon as a real, exercised feature of the library \u2014 not a half-finished
demo. The tests cover the contract ``Agent`` expects:

* ``choose`` ships the dict shape that ``agent.command('predict')``
  consumes, including ``choice`` / ``operation`` / ``target`` /
  ``probabilities`` / ``confidence`` / ``latency_ms``.
* DONE fires only when a goal keyword appears on the page, not just in
  the element table.
* TYPE_TEXT preferred over CLICK when the goal names a text-bearing field.
* ``field_text`` extracts a quoted phrase from the goal.

The tests monkey-patch the env var because the backend is selected at
call time, not at import time.
"""

from __future__ import annotations

import pytest

from my_autox_server import FakeAutoX
from my_autox_server.model import _choose_scripted, field_text


@pytest.fixture(autouse=True)
def scripted_env(monkeypatch):
    monkeypatch.setenv("AGENT_DECISION_BACKEND", "scripted")
    monkeypatch.setenv("AGENT_TEXT_BACKEND", "scripted")


def test_choose_returns_typesafe_shape():
    fake = FakeAutoX()
    page = fake.observe()
    decision = _choose_scripted(page, "Open Settings", [])
    for key in (
        "choice",
        "operation",
        "target",
        "confidence",
        "probabilities",
        "operation_probabilities",
        "latency_ms",
        "backend",
    ):
        assert key in decision, f"missing {key!r}"
    assert decision["backend"] == "scripted"
    # The first action in the scripted mock is the search textbox, but
    # ``Open Settings`` should land on the "Open Settings" button.
    assert decision["operation"] == "CLICK"


def test_done_fires_when_goal_keyword_already_on_page():
    fake = FakeAutoX()
    fake._state_name = "about"  # Pretend we're already on the About screen.
    page = fake.observe()
    decision = _choose_scripted(page, "Drill into About. Stop when shown.", [])
    assert decision["operation"] == "DONE"
    assert decision["choice"] == "DONE"


def test_done_requires_keyword_on_page_not_just_in_actions():
    fake = FakeAutoX()
    page = fake.observe()
    # "about" only appears in the *element label*, not the title/url/text.
    decision = _choose_scripted(page, "Find the about screen and stop when shown", [])
    assert decision["operation"] != "DONE"


def test_type_text_preferred_when_field_label_matches_goal():
    fake = FakeAutoX()
    page = fake.observe()
    decision = _choose_scripted(page, "Type 'iPhone 15 case' into the search products box", [])
    assert decision["operation"] == "TYPE_TEXT"
    # The first textbox in the home mock has the label "Search products".
    assert decision["target"] == "1"


def test_field_text_extracts_quoted_phrase():
    context = {
        "goal": "Type 'iPhone 15 case' into the search box",
        "field": {"label": "Search products", "role": "textbox", "value": ""},
        "page": {"title": "Home", "text": ""},
        "recent_actions": [],
    }
    value, helper = field_text(context)
    assert value == "iPhone 15 case"
    assert helper["model"] == "scripted"


def test_field_text_extracts_double_quoted_phrase():
    context = {
        "goal": 'Search for "red sneakers"',
        "field": {"label": "Search"},
        "page": {"title": "", "text": ""},
        "recent_actions": [],
    }
    value, _helper = field_text(context)
    assert value == "red sneakers"


def test_field_text_falls_back_to_last_goal_word():
    context = {
        "goal": "Type hello into the search box",
        "field": {"label": "Search"},
        "page": {"title": "", "text": ""},
        "recent_actions": [],
    }
    value, _helper = field_text(context)
    # No quote present; the helper picks the last non-stopword. "hello"
    # is the only real content word, so it wins.
    assert value == "hello"


def test_choose_agent_loop_walks_settings_to_about_with_fake():
    """End-to-end smoke: scripted backend + FakeAutoX walks home \u2192 about.

    This is the same path the verified demo will exercise against a
    real phone, but runs with no MCP server and no API key.
    """
    from my_autox_server import Agent

    fake = FakeAutoX()
    with Agent("demo://settings", "Open Settings, tap About, stop when shown.", device=fake) as agent:
        for _state in agent.run():
            pass
    assert "About" in agent.state["page"]["title"], (
        f"Final screen should be About; got title={agent.state['page']['title']!r}"
    )
    decisions = [d["operation"] for d in agent.state["decisions"]]
    assert "DONE" in decisions, f"expected DONE in decisions, got {decisions}"
    assert all(op != "BLOCKED" for op in decisions), f"loop blocked: {decisions}"

def test_choose_prefers_next_sibling_when_a_radio_is_checked():
    """The radio-group cycle rule: clicking the next sibling of the checked row.

    Without this rule, ``_choose_scripted`` would tie ``Silent`` and
    ``Vibrate`` (and ``Sound``) all at score 1 and ping-pong forever.
    The demo ``examples/toggle_silent_mode.py --fake`` depends on
    advancing exactly one slot each iteration; this test guards it.
    """
    from my_autox_server.autox import _mock_action_dict, _ring_mode_screen

    page_actions = [
        _mock_action_dict(m) for m in _ring_mode_screen("silent")["actions"]
    ]
    for index, action in enumerate(page_actions, start=1):
        action["id"] = f"e{index}"
        action["node"] = f"node-{action['label']}"
    page = {
        "url": "com.example.app/com.example.app.SoundActivity",
        "title": "Demo \u00b7 Sound",
        "text": "Sound\nSilent (selected), Vibrate, Sound",
        "actions": page_actions,
        "scroll": {"y": 0, "height": 2400},
        "package": "com.example.app",
        "activity": "com.example.app.SoundActivity",
    }
    decision = _choose_scripted(
        page,
        "Tap the next sound. Stop after the change is applied.",
        [],
    )
    assert decision["operation"] == "CLICK"
    # ``Silent`` is checked -> next sibling is ``Vibrate``.
    assert decision["choice"] == "e2", (
        f"expected click on Vibrate (e2); got {decision['choice']!r}"
    )


def test_choose_wraps_around_when_checked_radio_is_last():
    """Clicking past the end of the radio list wraps to the first sibling."""
    from my_autox_server.autox import _mock_action_dict, _ring_mode_screen

    page_actions = [
        _mock_action_dict(m) for m in _ring_mode_screen("sound")["actions"]
    ]
    for index, action in enumerate(page_actions, start=1):
        action["id"] = f"e{index}"
        action["node"] = f"node-{action['label']}"
    page = {
        "url": "com.example.app/com.example.app.SoundActivity",
        "title": "Demo \u00b7 Sound",
        "text": "Sound\nSilent, Vibrate, Sound (selected)",
        "actions": page_actions,
        "scroll": {"y": 0, "height": 2400},
        "package": "com.example.app",
        "activity": "com.example.app.SoundActivity",
    }
    decision = _choose_scripted(
        page,
        "Tap the next sound. Stop after the change is applied.",
        [],
    )
    assert decision["choice"] == "e1", (
        f"expected wrap to Silent (e1); got {decision['choice']!r}"
    )


def test_choose_does_not_cycle_when_no_radio_is_checked():
    """Without a checked row the cycle rule shouldn't fire."""
    from my_autox_server.autox import _mock_action_dict, _ring_mode_screen

    page_actions = [
        {**_mock_action_dict(m), "checked": "false"}
        for m in _ring_mode_screen("sound")["actions"]
    ]
    for index, action in enumerate(page_actions, start=1):
        action["id"] = f"e{index}"
        action["node"] = f"node-{action['label']}"
    page = {
        "url": "com.example.app/com.example.app.SoundActivity",
        "title": "Demo \u00b7 Sound",
        "text": "Sound\nSilent, Vibrate, Sound",
        "actions": page_actions,
        "scroll": {"y": 0, "height": 2400},
        "package": "com.example.app",
        "activity": "com.example.app.SoundActivity",
    }
    decision = _choose_scripted(page, "点击「响铃」。", [])
    # Without a checked row the cycle rule cannot fire, so the keyword
    # scorer decides. It must land on the label that matches the goal
    # (响铃 is the third radio, ``e3``) rather than blindly take the
    # first sibling (``e1``) the way the cycle rule would.
    assert decision["operation"] == "CLICK"
    assert decision["choice"] == "e3"


def test_goal_keywords_are_cjk_aware():
    """Chinese has no spaces; splitting must still yield matchable tokens."""
    from my_autox_server.model import _goal_keywords

    tokens = _goal_keywords("点击「响铃」。")
    # The whole run, its characters, and its bigrams all appear, so a
    # label like "响铃" is reachable.
    assert "响铃" in tokens
    assert "响" in tokens and "铃" in tokens

    latin = _goal_keywords("Open Settings, tap About.")
    assert latin == ["open", "settings", "tap", "about"]


def test_scripted_backend_matches_chinese_labels():
    from my_autox_server.autox import _mock_action_dict, _ring_mode_screen

    page_actions = [_mock_action_dict(m) for m in _ring_mode_screen("silent")["actions"]]
    for index, action in enumerate(page_actions, start=1):
        action["id"] = f"e{index}"
        action["node"] = f"node-{action['label']}"
    page = {
        "url": "com.example.app/com.example.app.SoundActivity",
        "title": "Demo · Sound",
        "text": "声音和振动\n声音模式\n静音（已选中）, 振动, 响铃",
        "actions": page_actions,
        "scroll": {"y": 0, "height": 2400},
        "package": "com.example.app",
        "activity": "com.example.app.SoundActivity",
    }
    # 静音 is checked, so the cycle rule fires and targets the next
    # sibling (振动 = e2) regardless of the goal's wording.
    decision = _choose_scripted(page, "把它切换成「振动」。", [])
    assert decision["operation"] == "CLICK"
    assert decision["choice"] == "e2"
