"""Verify that the auxiliary-LLM planner's ``RELEVANT_APPS=…`` line
narrows the ``launch_<pkg>`` actions exposed to Jev.

Before this change the planner returned ``RELEVANT_APPS=闲鱼`` for the
goal "打开闲鱼, 切换到 西安, 搜索跑步机" but the Jev payload still
listed every installed app's ``LAUNCH_<PKG>`` operation plus every
launch_<pkg> entry in ``launch_target.criteria``. The agent would
spend probability mass on 50+ irrelevant apps, picking the right one
mostly by accident.

After this change ``_choose_typesafe`` (and the scripted backend) drop
every launch entry whose label doesn't match any relevant app, so the
Jev payload contains only the planner-acknowledged apps.
"""

from __future__ import annotations

import os
from unittest import mock

# Use the scripted backend so the test exercises the production filter
# without a TypeSafe round-trip; we separately monkey-patch the
# typesafe path to capture the criteria it builds.
os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
os.environ.setdefault("TYPESAFE_API_KEY", "test-fake-key")

from mobile_jev_ultrafast.autox import _build_actions, _launch_actions
from mobile_jev_ultrafast.model import (
    _filter_launch_choices,
    _strip_launch_prefix,
    action_space,
    choose,
)


# A representative slice of an installed-app list. ``_launch_actions``
# adds these on top of the always-available :data:`_KNOWN_APPS` list.
_INSTALLED = [
    {"package": "com.taobao.idlefish", "label": "闲鱼"},
    {"package": "com.taobao.taobao", "label": "淘宝"},
    {"package": "com.tencent.mm", "label": "微信"},
    {"package": "com.xunmeng.pinduoduo", "label": "拼多多"},
    {"package": "com.ss.android.ugc.aweme", "label": "抖音"},
    {"package": "com.autox.app", "label": "Autox.js v7"},
]


def _build_launch_choices():
    """Run the production action builder so the test sees the same
    label prefixes (``启动 <label>``) the real loop uses."""
    return {
        action["id"]: {
            "label": action["label"],
            "package": action["value"],
            "role": "app",
        }
        for action in _launch_actions(_INSTALLED)
        if action["kind"] == "launch"
    }


def _build_full_action_table():
    """Full action table: launch entries + the synthetic controls
    (PRESS_HOME / PRESS_BACK / SWIPE_* / SCROLL_* / WAIT / DOUBLE_TAP).
    Mirrors what ``AutoX.observe`` produces."""
    return _build_actions(
        {"c": "FrameLayout", "b": [0, 0, 100, 100], "a": "", "children": []},
        launch_apps=_INSTALLED,
    )


def test_strip_launch_prefix_handles_zh_and_en():
    assert _strip_launch_prefix("启动 闲鱼") == "闲鱼"
    assert _strip_launch_prefix("Launch WeChat") == "WeChat"
    assert _strip_launch_prefix("按 Home 键回到桌面") == "按 Home 键回到桌面"
    assert _strip_launch_prefix("") == ""


def test_filter_keeps_only_relevant_apps():
    """Goal = 闲鱼 → launch_target only shows the one relevant app."""
    launch_choices = _build_launch_choices()
    filtered = _filter_launch_choices(launch_choices, ["闲鱼"])
    labels = sorted(_strip_launch_prefix(c["label"]) for c in filtered.values())
    # Only the 闲鱼 entry should survive. The matching is exact-substring
    # so 微信 / 淘宝 / 拼多多 / 抖音 / etc. all get pruned.
    assert labels == ["闲鱼"], labels


def test_filter_keeps_all_apps_when_relevant_apps_empty():
    """No relevant_apps → preserve old behaviour (all launches visible)."""
    launch_choices = _build_launch_choices()
    for empty in (None, [], ["", "   "]):
        filtered = _filter_launch_choices(launch_choices, empty)
        assert set(filtered) == set(launch_choices), empty


def test_filter_matches_parent_brand_substring():
    """Relevant label appearing as a substring inside the launch label
    should still match — e.g. ``淘宝`` matches ``启动 闲鱼 (淘宝)``.
    """
    labelled = {
        "launch_a": {"label": "启动 闲鱼 (淘宝二手)", "package": "com.taobao.idlefish", "role": "app"},
        "launch_b": {"label": "启动 微信", "package": "com.tencent.mm", "role": "app"},
    }
    filtered = _filter_launch_choices(labelled, ["淘宝"])
    assert set(filtered) == {"launch_a"}


def test_filter_falls_back_to_all_when_no_matches():
    """If the LLM mislabelled every relevant app, refuse to brick the
    loop — keep the full set so the agent can still launch anything
    visible on screen."""
    launch_choices = _build_launch_choices()
    filtered = _filter_launch_choices(launch_choices, ["NonExistentApp"])
    assert set(filtered) == set(launch_choices)
    assert len(filtered) == len(launch_choices)


def test_choose_threads_relevant_apps_to_backend():
    """The high-level dispatch should pass ``relevant_apps`` through so
    the typesafe backend also sees the narrowed set. We capture the
    launch_target.criteria the backend constructs by spying on
    ``post_json``."""
    from mobile_jev_ultrafast import model as model_module

    captured: dict = {}

    def fake_post_json(url, key, body):
        captured["criteria"] = body["questions"]["launch_target"]["criteria"]
        # Echo the first launch target back as a valid TypeSafe answer
        # so validate_choice() passes.
        op_ids = list(body["questions"]["operation"]["criteria"])
        target_ids = list(captured["criteria"])
        return {
            "answers": {
                "operation": {
                    "choice": "LAUNCH_APP",
                    "probabilities": {op: (1.0 if op == "LAUNCH_APP" else 0.0) for op in op_ids},
                    "confidence": 0.9,
                },
                "task_complete": {
                    "choice": "continue",
                    "probabilities": {"continue": 1.0, "finish": 0.0},
                    "confidence": 1.0,
                },
                "launch_target": {
                    "choice": target_ids[0],
                    "probabilities": {tid: (1.0 if tid == target_ids[0] else 0.0) for tid in target_ids},
                    "confidence": 0.9,
                },
            },
            "model": "fake",
            "usage": {},
        }

    # Build the action table the loop would see on the launcher.
    actions = _launch_actions(_INSTALLED)

    with mock.patch.dict(os.environ, {"AGENT_DECISION_BACKEND": "typesafe"}):
        with mock.patch.object(model_module, "post_json", fake_post_json):
            choose(
                {
                    "url": "launcher",
                    "title": "launcher",
                    "text": "",
                    "actions": actions,
                    "fingerprint": "x",
                },
                "打开闲鱼, 切换到 西安, 搜索跑步机",
                [],
                relevant_apps=["闲鱼"],
            )

    criteria = captured["criteria"]
    labels = sorted(_strip_launch_prefix(c["element"]) for c in criteria.values())
    # Only the 闲鱼 entry should be exposed as a launch_target. The
    # _KNOWN_APPS list includes some unrelated apps (设置/微信/QQ/淘宝/
    # 抖音/飞书/...); the filter must drop every one of them.
    assert labels == ["闲鱼"], labels


def test_choose_without_relevant_apps_keeps_full_launch_surface():
    """When ``relevant_apps`` is ``None`` we preserve the legacy
    behaviour so plans without the planner (--no-plan) still work."""
    from mobile_jev_ultrafast import model as model_module

    captured: dict = {}

    def fake_post_json(url, key, body):
        captured["criteria"] = body["questions"]["launch_target"]["criteria"]
        op_ids = list(body["questions"]["operation"]["criteria"])
        target_ids = list(captured["criteria"])
        return {
            "answers": {
                "operation": {
                    "choice": "LAUNCH_APP",
                    "probabilities": {op: (1.0 if op == "LAUNCH_APP" else 0.0) for op in op_ids},
                    "confidence": 0.9,
                },
                "task_complete": {
                    "choice": "continue",
                    "probabilities": {"continue": 1.0, "finish": 0.0},
                    "confidence": 1.0,
                },
                "launch_target": {
                    "choice": target_ids[0],
                    "probabilities": {tid: (1.0 if tid == target_ids[0] else 0.0) for tid in target_ids},
                    "confidence": 0.9,
                },
            },
            "model": "fake",
            "usage": {},
        }

    actions = _launch_actions(_INSTALLED)

    with mock.patch.dict(os.environ, {"AGENT_DECISION_BACKEND": "typesafe"}):
        with mock.patch.object(model_module, "post_json", fake_post_json):
            choose(
                {
                    "url": "launcher",
                    "title": "launcher",
                    "text": "",
                    "actions": actions,
                    "fingerprint": "x",
                },
                "打开闲鱼, 切换到 西安, 搜索跑步机",
                [],
            )

    labels = sorted(_strip_launch_prefix(c["element"]) for c in captured["criteria"].values())
    # Every launch entry should be present — the legacy behaviour must
    # survive untouched when the planner is disabled.
    assert "闲鱼" in labels
    assert "微信" in labels
    assert "淘宝" in labels
    assert "抖音" in labels
    assert len(labels) >= len({a["value"] for a in actions if a["kind"] == "launch"})


def test_choose_filters_operation_criteria_not_just_launch_target():
    """The ``LAUNCH_<PKG>`` synthetic controls in ``operations`` must
    also be pruned so the model's operation-level probability vector
    doesn't waste mass on irrelevant apps."""
    from mobile_jev_ultrafast import model as model_module

    captured: dict = {}

    def fake_post_json(url, key, body):
        captured["operation_criteria"] = body["questions"]["operation"]["criteria"]
        op_ids = list(captured["operation_criteria"])
        return {
            "answers": {
                "operation": {
                    "choice": "BLOCKED",
                    "probabilities": {op: (1.0 if op == "BLOCKED" else 0.0) for op in op_ids},
                    "confidence": 0.9,
                },
                "task_complete": {
                    "choice": "finish",
                    "probabilities": {"continue": 0.0, "finish": 1.0},
                    "confidence": 1.0,
                },
            },
            "model": "fake",
            "usage": {},
        }

    actions = _build_full_action_table()

    with mock.patch.dict(os.environ, {"AGENT_DECISION_BACKEND": "typesafe"}):
        with mock.patch.object(model_module, "post_json", fake_post_json):
            choose(
                {
                    "url": "launcher",
                    "title": "launcher",
                    "text": "",
                    "actions": actions,
                    "fingerprint": "x",
                },
                "打开闲鱼",
                [],
                relevant_apps=["闲鱼"],
            )

    op_criteria = captured["operation_criteria"]
    # Synthesised LAUNCH_<PKG> entries (e.g. ``LAUNCH_COM.TAOBAO.IDLEFISH``)
    # are emitted by :func:`action_space` with their id upper-cased.
    # ``LAUNCH_APP`` is the abstract operation key, separate from the
    # per-package ``LAUNCH_<PKG>`` controls.
    package_launch_ops = [k for k in op_criteria if k.startswith("LAUNCH_") and k != "LAUNCH_APP"]
    assert package_launch_ops == ["LAUNCH_COM.TAOBAO.IDLEFISH"], package_launch_ops
    # The abstract LAUNCH_APP operation stays — its description is
    # independent of how many launch entries exist.
    assert "LAUNCH_APP" in op_criteria
    # Non-launch controls must survive intact.
    for required in ("PRESS_HOME", "PRESS_BACK", "SWIPE_LEFT", "SCROLL_UP", "DONE", "BLOCKED"):
        assert required in op_criteria, f"missing synthetic control {required}"


def test_action_space_builds_full_launch_set_for_baseline():
    """Sanity check: the unfiltered launch table the loop sees must
    contain the apps we expect so the filter actually has work to do."""
    actions = _launch_actions(_INSTALLED)
    elements, _, controls, launch_choices = action_space(actions)
    packages = {c["package"] for c in launch_choices.values()}
    assert {"com.taobao.idlefish", "com.tencent.mm", "com.taobao.taobao"} <= packages
    # The corresponding ``LAUNCH_<PKG>`` controls exist for every launch.
    launch_controls = {cid for cid in controls if cid.lower().startswith("launch_")}
    assert launch_controls  # non-empty