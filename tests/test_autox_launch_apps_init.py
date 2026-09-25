"""Regression test: AutoX.__init__ must not drop the LAUNCH_APP surface.

History
-------
The auxiliary-LLM planner returns ``RELEVANT_APPS=闲鱼`` for the goal
"打开闲鱼, 切换到 西安, 搜索跑步机", but the Jev payload still listed
every curated ``_KNOWN_APPS`` entry (``启动 设置``, ``Launch Gmail``,
``Launch YouTube`` etc.) instead of the user's installed apps. Root
cause: ``AutoX.__init__`` ran ``_ensure_installed_apps()`` which set
``self.launch_apps`` to the MCP-detected list, then a few lines later
re-assigned ``self.launch_apps`` to ``[]`` — silently nuking every
installed app before ``_build_actions`` ran. The filter at the Jev
level then matched 0 entries and fell back to the curated 23, so the
user saw "the LLM returned the right apps but the options still show
all the unrelated ones".

This test pins the contract: after construction, ``AutoX.launch_apps``
must be the post-filter installed-app list (NOT ``[]``), and a
``_build_actions`` call must surface those packages as
``launch_<package>`` actions.
"""

from __future__ import annotations

import os

os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
os.environ.setdefault("TYPESAFE_API_KEY", "test-fake-key")

from unittest import mock

from mobile_jev_ultrafast.autox import AutoX, _build_actions


class _StubMCP:
    """Minimal MCP stand-in that returns a UI tree + installed apps.

    ``installed_apps`` deliberately includes 闲鱼 (a non-system package
    that is NOT in the curated ``_KNOWN_APPS`` list) so the test can
    prove the launch surface actually came from the device probe.
    """

    def __init__(self) -> None:
        self.display = (1080, 2400)

    def display_size(self):
        return self.display

    def get_ui_tree(self):
        return {"c": "FrameLayout", "b": [0, 0, 1080, 2400], "a": "", "children": []}

    def screenshot(self, as_base64=False):
        return ""

    def installed_apps(self):
        return [
            {"label": "闲鱼", "package": "com.taobao.idlefish", "system": False},
            {"label": "微信", "package": "com.tencent.mm", "system": False},
            {"label": "淘宝", "package": "com.taobao.taobao", "system": False},
            {"label": "高德地图", "package": "com.autonavi.minimap", "system": False},
        ]


def test_autox_init_preserves_launch_apps():
    """``AutoX.launch_apps`` must reflect the MCP probe result.

    The bug was a stray ``self.launch_apps: list[dict] = []`` line that
    ran AFTER ``_ensure_installed_apps()`` and clobbered the populated
    list. The fix moves the attribute declaration above the probe; this
    test fails loudly if anyone re-introduces the overwrite.
    """
    with mock.patch.dict(os.environ, {"LAUNCH_APP_ALLOWLIST": "", "AUTOX_MCP_URL": "stub"}):
        device = AutoX(mcp=_StubMCP())
    try:
        # The MCP probe returned 4 installed apps; none are in the
        # curated _KNOWN_APPS list so this assertion is unambiguous.
        assert len(device.installed_apps) == 4, device.installed_apps
        assert len(device.launch_apps) == 4, device.launch_apps
        packages = {row["package"] for row in device.launch_apps}
        assert "com.taobao.idlefish" in packages  # 闲鱼
        assert "com.tencent.mm" in packages       # 微信
    finally:
        device.close()


def test_build_actions_surfaces_installed_apps():
    """The action table must expose installed apps as launch_* entries.

    Before the fix, only the 23 curated ``_KNOWN_APPS`` entries made it
    into ``launch_choices`` — the filter at the Jev level then matched
    0 relevant apps and fell back to the same 23, so the model never
    saw 闲鱼 / 微信 / 淘宝 as launch options even though they were
    detected as installed.
    """
    with mock.patch.dict(os.environ, {"LAUNCH_APP_ALLOWLIST": "", "AUTOX_MCP_URL": "stub"}):
        device = AutoX(mcp=_StubMCP())
    try:
        actions = _build_actions(
            {"c": "FrameLayout", "b": [0, 0, 1080, 2400], "a": "", "children": []},
            launch_apps=device.launch_apps,
        )
        launch_action_ids = [a["id"] for a in actions if a["kind"] == "launch"]
        assert "launch_com.taobao.idlefish" in launch_action_ids
        assert "launch_com.tencent.mm" in launch_action_ids
        assert "launch_com.taobao.taobao" in launch_action_ids
        assert "launch_com.autonavi.minimap" in launch_action_ids
    finally:
        device.close()


def test_autox_init_with_probe_failure_falls_back_via_build_actions():
    """When the MCP probe fails, ``launch_apps`` is ``[]`` but
    :func:`_build_actions` still surfaces the curated ``_KNOWN_APPS``
    fallback so Jev has at least the system/common apps to launch.

    This codifies the post-fix contract: ``launch_apps`` is the
    installed-app probe result (possibly empty), and the curated
    fallback lives in :func:`_launch_actions` regardless of whether the
    probe succeeded. The buggy overwrite made BOTH paths disappear
    because the whole ``_launch_actions(launch_apps)`` call would still
    work — the bug only manifested when the probe DID succeed.
    """
    class _BrokenMCP(_StubMCP):
        def installed_apps(self):
            raise RuntimeError("list_apps tool unavailable on this build")

    with mock.patch.dict(os.environ, {"LAUNCH_APP_ALLOWLIST": "", "AUTOX_MCP_URL": "stub"}):
        device = AutoX(mcp=_BrokenMCP())
    try:
        assert device.installed_apps == []
        assert device.launch_apps == []
        # The curated fallback is layered in _build_actions /
        # _launch_actions on top of the empty probe result.
        actions = _build_actions(
            {"c": "FrameLayout", "b": [0, 0, 1080, 2400], "a": "", "children": []},
            launch_apps=device.launch_apps,
        )
        launch_ids = [a["id"] for a in actions if a["kind"] == "launch"]
        assert "launch_com.android.settings" in launch_ids
        assert "launch_com.tencent.mm" in launch_ids
    finally:
        device.close()