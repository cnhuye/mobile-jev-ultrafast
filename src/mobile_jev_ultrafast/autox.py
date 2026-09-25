"""Device layer.

Mirrors :mod:`jev_ultrafast.browser`'s public API (``observe`` / ``fresh`` /
``act`` / ``close``) so :mod:`.agent` stays a one-for-one port. The
browser harness over CDP is replaced by an AutoX.js client over MCP.

Two implementations live here:

* :class:`AutoX` — connects to the phone via
  :class:`mobile_jev_ultrafast.mcp_client.MCPClient`.
* :class:`FakeAutoX` — returns a hand-crafted mock so the inspector,
  agent loop, and unit tests run without a device on the network. Set
  ``AGENT_USE_FAKE=1`` (or pass ``fake=True``) to pick this backend.

UI tree fields are mapped per ``../docs/MCP_USAGE.md`` §"get_ui_tree":
``c`` (class), ``id`` (resource-id), ``t`` (text), ``d`` (content-desc),
``b`` (bounds as ``[x1,y1,x2,y2]``), ``a`` (interaction flags as a
``c,f,s,l,d`` string), and ``children``. Older builds send verbose class
names like ``android.widget.Button``; the classifier accepts both.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass

from .mcp_client import MCPClient


class StalePage(ValueError):
    """A decision no longer refers to the observed screen."""


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

# 1×1 black PNG; inspector stretches it to the reported screen size.
_BLANK_SCREENSHOT = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


# Default swipe duration (ms). 400ms gives Android's GestureDescription a
# ``distance/duration`` ratio that survives both fling-style RecyclerView
# scrollers and Compose lazy lists — the original 300ms sometimes
# triggers fling and gets ignored by lazy columns. Tunable via
# ``AUTOX_SWIPE_DURATION``.
_SWIPE_DURATION_DEFAULT_MS = 400


def _swipe_duration_ms() -> int:
    """Resolve the swipe duration from ``AUTOX_SWIPE_DURATION``.

    Accepts positive integers in milliseconds. Falls back to
    :data:`_SWIPE_DURATION_DEFAULT_MS` for unset / malformed values
    rather than crashing the act loop.
    """
    raw = os.environ.get("AUTOX_SWIPE_DURATION")
    if not raw:
        return _SWIPE_DURATION_DEFAULT_MS
    try:
        ms = int(raw)
    except (TypeError, ValueError):
        return _SWIPE_DURATION_DEFAULT_MS
    return ms if ms > 0 else _SWIPE_DURATION_DEFAULT_MS


# AutoX's compact UI tree nests children under each node. The walker
# yields every node so ``_build_actions`` / ``_build_text`` can iterate
# in document order without juggling parent pointers.
def _walk(node):
    yield node
    for child in node.get("children") or []:
        yield from _walk(child)


# Short aliases that appear in either compact format (``Button``) or
# the older verbose format (``android.widget.Button``).
_ROLE_TABLE = {
    "button": "button",
    "imagebutton": "button",
    "checkbox": "checkbox",
    "radiobutton": "radio",
    "switch": "switch",
    "togglebutton": "switch",
    "edittext": "textbox",
    "autocompletetextview": "textbox",
    "multiautocompletetextview": "textbox",
    "textview": "text",
    "spinner": "combobox",
    "listview": "list",
    "recyclerview": "list",
    "nestedscrollview": "view",
    "scrollview": "view",
}


def _short_class(node) -> str:
    """Lower-cased short class name (``android.widget.Button`` -> ``button``)."""
    c = (node.get("c") or "").strip().lower()
    if not c:
        return ""
    return c.split(".")[-1]


def _classify(node) -> str:
    short = _short_class(node)
    if short in _ROLE_TABLE:
        return _ROLE_TABLE[short]
    # Layout containers with clickable / checkable descendants show up as
    # flat rows on some builds; treat them as buttons so the agent sees
    # them instead of an anonymous "linearlayout".
    a = _interaction_flags(node)
    if ("c" in a or "k" in a) and short.startswith(("linear", "relative", "frame", "constraint")):
        return "button"
    return short or "view"


# ``a`` is a compact comma/space-separated string of single-character
# flags. ``c`` = clickable, ``f`` = focusable, ``s`` = scrollable,
# ``l`` = long-clickable, ``d`` = disabled. Some builds also send the
# verbose ``clickable=true`` style; we normalise both.
_INTERACTION_TOKENS = {
    # Compact single-character symbols (per MCP_USAGE.md).
    "c": "c",
    "f": "f",
    "s": "s",
    "l": "l",
    "d": "d",
    # ``k`` = checkable (checkbox / radio / switch), ``x`` = selected.
    # These two come from the newer MCP builds that expose the checkable
    # state; older builds simply don't emit them and the code degrades.
    "k": "k",
    "x": "x",
    # Legacy verbose forms.
    "click": "c",
    "clickable": "c",
    "long-click": "l",
    "longclick": "l",
    "long-clickable": "l",
    # The un-hyphenated adjective form. Without this entry the compact
    # fallback sees the letters l/c/k inside the word and mis-expands it
    # to ``"lck"``.
    "longclickable": "l",
    # Same trap for the other adjective forms that contain flag letters.
    "editable": "f",
    "focused": "f",
    "dismissable": "l",
    "focus": "f",
    "focusable": "f",
    "edit": "f",
    "scroll": "s",
    "scrollable": "s",
    "disabled": "d",
    "checkable": "k",
    "selected": "x",
}


def _norm_checked(value) -> str | None:
    """Normalise a JSON checked value to ``"true"`` / ``"false"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on", "checked"):
        return "true"
    if text in ("false", "0", "no", "off", "unchecked"):
        return "false"
    return None


def _parent_map(root: dict) -> dict:
    """Map ``id(node)`` -> parent node for every node in the tree."""
    parents: dict = {}

    def visit(node):
        for child in node.get("children") or []:
            parents[id(child)] = node
            visit(child)

    visit(root)
    return parents


def _sibling_checked(node, parent) -> str | None:
    """Find a checked state for ``node``, looking at neighbours if needed.

    Android renders a radio/checkbox row as a clickable container holding
    a label ``TextView`` and a state-bearing ``RadioButton``. The state
    lives on the latter, but the *label* the policy reasons about is on
    the former, so we propagate the state within the row. Without this
    the model sees "响铃 / 振动 / 静音" as three indistinguishable
    options — see ``docs/REQUIREMENTS_MY_AUTOX.md`` for why the server
    side needs to emit ``checked`` in the first place.

    Lookup order: the node itself, then its siblings (for the label
    ``TextView``), then any descendant (for the clickable container).
    Descendants are only searched when the node owns no text of its own,
    so a list container doesn't inherit an unrelated child's state.
    """
    own = _norm_checked(node.get("checked"))
    if own is not None:
        return own
    if parent is not None:
        for sibling in parent.get("children") or []:
            if sibling is node:
                continue
            value = _norm_checked(sibling.get("checked"))
            if value is not None:
                return value
    # Containers (no text of their own) inherit from the first checkable
    # descendant — this is the clickable row case.
    if not (node.get("t") or node.get("d")):
        for child in _walk(node):
            if child is node:
                continue
            value = _norm_checked(child.get("checked"))
            if value is not None:
                return value
    return None


# The compact form is a *concatenation* of single characters (the real
# device emits ``a: "cf"``, ``a: "fs"``, ``a: "k"``), while older builds
# emit ``key=value`` tokens (``clickable=true``). Both must parse.
_SINGLE_CHAR_FLAGS = "cfsldkx"


def _interaction_flags(node) -> str:
    """Return a compact string of single-character interaction flags.

    Handles all three shapes seen in the wild:

    * concatenated compact: ``"cf"``, ``"cfs"``, ``"kd"`` (real device)
    * separated compact:    ``"c, f"`` / ``"c,f"``
    * legacy verbose:       ``"clickable=true, focusable=true"``
    """
    raw = (node.get("a") or "").lower().replace(",", " ").split()
    flags: list[str] = []
    for token in raw:
        # Strip ``key=value`` shape, e.g. ``clickable=true``.
        key = token.split("=", 1)[0].strip()
        if not key:
            continue
        # 1) A known verbose word (``clickable``, ``edit``, ``longclickable``…)
        #    maps to exactly one flag. Check this first so dictionary words
        #    that happen to contain flag letters (``edit`` has a ``d``,
        #    ``scrollable`` has a ``c``) aren't mistaken for compact tokens.
        verbose = _INTERACTION_TOKENS.get(key)
        if verbose is not None:
            if verbose not in flags:
                flags.append(verbose)
            continue
        # 2) Otherwise treat it as compact (possibly concatenated) and
        #    keep every flag character, ignoring unknown ones. This covers
        #    ``"cf"``, ``"fs"``, ``"k"`` and oddities like ``"ca"``.
        for ch in key:
            if ch in _SINGLE_CHAR_FLAGS and ch not in flags:
                flags.append(ch)
    return "".join(flags)


def _bounds_rect(bounds):
    """``[x1, y1, x2, y2]`` -> ``{x, y, w, h}``."""
    b = bounds or [0, 0, 0, 0]
    if len(b) < 4:
        b = list(b) + [0] * (4 - len(b))
    x1, y1, x2, y2 = b[:4]
    return {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)}


def _bounds_center(bounds):
    b = bounds or [0, 0, 0, 0]
    if len(b) < 4:
        b = list(b) + [0] * (4 - len(b))
    x1, y1, x2, y2 = b[:4]
    return (x1 + x2) // 2, (y1 + y2) // 2


def _node_id(node) -> str:
    """Stable id per (bounds, class, text, desc). Android has no DOM
    references; bounds is the closest analog to ``getBoundingClientRect``
    (``../docs/CLIENT_INTEGRATION.md`` §5.1)."""
    payload = json.dumps(
        [
            node.get("b"),
            node.get("c"),
            node.get("t") or "",
            node.get("d") or "",
            node.get("id") or "",
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _node_kind(node, role: str) -> str | None:
    """Decide if the action on this node is click, fill, or select."""
    flags = _interaction_flags(node)
    if role == "combobox":
        return "select"
    if role == "textbox":
        return "fill"
    # Scrollable-only containers should be traversed via ``scroll_down`` /
    # ``scroll_up`` rather than as click targets.
    if role in {"list", "scrollview", "recyclerview"} and "c" not in flags and "l" not in flags:
        return None
    # ``text`` rows with visible text usually represent clickable list
    # items even when the device's UI Automator forgot to flag them.
    # We lean on the visible text instead of the missing clickable bit.
    if role == "text" and node.get("t"):
        return "click"
    return "click"


def _label_for(node) -> str:
    """Compose a stable label. Priority: text > content-desc > class."""
    label = node.get("t") or node.get("d") or _short_class(node) or "element"
    return str(label).strip() or "element"


def _descendant_label(node, *, depth: int = 0) -> str:
    """First non-empty text/desc inside this node's subtree.

    A clickable row usually has no ``t`` of its own — the label lives in
    a child ``TextView``. Falls back to ``""`` when the subtree has no
    text at all (then the caller keeps the class-name label).
    """
    if depth > 6:
        return ""
    for child in node.get("children") or []:
        own = (child.get("t") or child.get("d") or "").strip()
        if own:
            return own
        nested = _descendant_label(child, depth=depth + 1)
        if nested:
            return nested
    return ""


def _sibling_label(node, parent) -> str:
    """First non-empty text/desc among this node's siblings.

    A ``RadioButton`` carries the state but no text; the visible label is
    its sibling ``TextView``. Without this the element table would show
    three identical ``radiobutton`` rows.
    """
    if parent is None:
        return ""
    for sibling in parent.get("children") or []:
        if sibling is node:
            continue
        text = (sibling.get("t") or sibling.get("d") or "").strip()
        if text:
            return text
    return ""


def _build_actions(ui_tree: dict, *, launch_apps: list[dict] | None = None) -> list:
    actions: list = []
    counter = 0
    seen_bounds: set = set()
    parents = _parent_map(ui_tree)
    for raw in _walk(ui_tree):
        role = _classify(raw)
        a_value = raw.get("a", "") or ""
        has_text = bool(raw.get("t"))
        # Skip pure layout containers with no interactions and no text.
        # Keep checkable / selected nodes even when the build forgot to
        # emit an ``a`` flag for them.
        if not a_value and not (role == "text" and has_text) and raw.get("checked") is None:
            continue
        kind = _node_kind(raw, role)
        if kind is None:
            continue
        # Many Android UIs repeat the parent bounds on the child.
        # De-duplicate so the inspector doesn't show the same row twice.
        bkey = tuple(raw.get("b") or ())
        if bkey and bkey in seen_bounds and role in {"view", "linear", "relative", "frame"}:
            continue
        if bkey and role == "view":
            seen_bounds.add(bkey)
        counter += 1
        b = raw.get("b") or [0, 0, 0, 0]
        label = _label_for(raw)
        # A node whose label is just its class name inherits a real label:
        # containers take it from a labelled descendant; a checkable node
        # (``RadioButton``) takes it from a labelled sibling. This is what
        # makes the Sound-mode rows show up as "响铃 / 振动 / 静音"
        # instead of anonymous "linearlayout" / "radiobutton" entries.
        if label == _short_class(raw) or label == "element":
            inherit = _descendant_label(raw)
            if not inherit:
                inherit = _sibling_label(raw, parents.get(id(raw)))
            if inherit:
                label = inherit
        node_id = _node_id(raw)
        action = {
            "id": f"e{counter}",
            "kind": kind,
            "role": role,
            "label": label,
            "value": raw.get("t") or "",
            "node": node_id,
            "rect": _bounds_rect(b),
        }
        checked = _sibling_checked(raw, parents.get(id(raw)))
        if checked is not None:
            action["checked"] = checked
        if _norm_checked(raw.get("selected")) == "true":
            action["selected"] = "true"
        if raw.get("enabled") is False:
            action["enabled"] = "false"
        if role == "combobox":
            action["current_value"] = raw.get("t") or ""
        actions.append(action)
        # TextBoxes are also focusable; surface a click action so the policy
        # can deliberately focus without typing (e.g., open date picker).
        if kind == "fill":
            counter += 1
            actions.append(
                {
                    "id": f"e{counter}",
                    "kind": "click",
                    "role": role,
                    "label": f"Open {label}",
                    "value": raw.get("t") or "",
                    "node": node_id,
                    "rect": _bounds_rect(b),
                }
            )
    actions.extend(_synthetic_actions(launch_apps=launch_apps))
    return actions


# Curated list of system / common apps that are almost always useful on a
# stock Android phone. Surfaced as ``launch_<package>`` synthetic actions
# on every observation so Jev can call LAUNCH_APP without first having
# to enumerate installed apps. The full installed-app list is added on
# top when ``AutoX.launch_apps`` is populated (see ``_build_actions``).
#
# Each entry is ``(package_name, label_zh, label_en)``. We expose both
# labels so Jev can pick by either language; the match is keyword-based
# against the goal string at action-build time.
_KNOWN_APPS: list[tuple[str, str, str]] = [
    ("com.android.settings", "设置", "Settings"),
    ("com.android.dialer", "电话", "Phone"),
    ("com.android.contacts", "联系人", "Contacts"),
    ("com.android.messaging", "短信", "Messages"),
    ("com.android.camera", "相机", "Camera"),
    ("com.android.deskclock", "时钟", "Clock"),
    ("com.android.calendar", "日历", "Calendar"),
    ("com.android.browser", "浏览器", "Browser"),
    ("com.android.gallery3d", "相册", "Gallery"),
    ("com.android.documentsui", "文件", "Files"),
    ("com.android.systemui", "系统界面", "System UI"),
    ("com.android.launcher3", "桌面", "Launcher"),
    # Tencent
    ("com.tencent.mm", "微信", "WeChat"),
    ("com.tencent.mobileqq", "QQ", "QQ"),
    # Alibaba
    ("com.taobao.taobao", "淘宝", "Taobao"),
    ("com.eg.android.AlipayGphone", "支付宝", "Alipay"),
    ("com.alibaba.intl.android.apps.poseidon", "速卖通", "AliExpress"),
    # ByteDance
    ("com.ss.android.ugc.aweme", "抖音", "Douyin"),
    ("com.ss.android.lark", "飞书", "Lark"),
    # Google (if installed)
    ("com.google.android.apps.maps", "地图", "Maps"),
    ("com.google.android.gm", "Gmail", "Gmail"),
    ("com.google.android.youtube", "YouTube", "YouTube"),
    ("com.google.android.apps.photos", "相册", "Photos"),
]


def _launch_actions(extra_apps: list[dict] | None = None) -> list:
    """Build the LAUNCH_APP synthetic action table.

    ``extra_apps`` (optional) merges installed apps pulled from the phone
    via :meth:`MCPClient.installed_apps`. Each entry adds one
    ``launch_<package>`` action whose label is the app's display label.
    Duplicates against :data:`_KNOWN_APPS` are dropped by package name.
    """
    actions: list = []
    seen: set[str] = set()

    def _add(pkg: str, label: str) -> None:
        if not pkg or pkg in seen:
            return
        seen.add(pkg)
        actions.append(
            {
                "id": f"launch_{pkg}",
                "kind": "launch",
                "role": "app",
                "label": f"启动 {label}" if re.search(r"[\u4e00-\u9fff]", label) else f"Launch {label}",
                "value": pkg,
                "node": f"launch-{pkg}",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            }
        )

    for pkg, label_zh, label_en in _KNOWN_APPS:
        _add(pkg, label_zh or label_en)
    if extra_apps:
        for row in extra_apps:
            pkg = (row.get("package") or "").strip()
            label = (row.get("label") or "").strip()
            if not pkg or row.get("system"):
                continue
            _add(pkg, label or pkg)
    return actions


def _synthetic_actions(launch_apps: list[dict] | None = None) -> list:
    """The always-available controls.

    Order matters: model sees them as candidates for ``BLOCKED``
    fallback, so the most useful one (LAUNCH_APP) is listed first.
    """
    actions: list = list(_launch_actions(launch_apps))
    actions.extend(
        [
            {
                "id": "press_home",
                "kind": "key",
                "label": "按 Home 键回到桌面",
                "node": -1,
                "role": "key",
                "key": "home",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "press_back",
                "kind": "key",
                "label": "按返回键",
                "node": -1,
                "role": "key",
                "key": "back",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "press_recents",
                "kind": "key",
                "label": "按最近任务键",
                "node": -1,
                "role": "key",
                "key": "recents",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "swipe_left",
                "kind": "swipe",
                "label": "向左滑动（桌面/列表翻页）",
                "direction": "left",
                "node": -1,
                "role": "swipe",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "swipe_right",
                "kind": "swipe",
                "label": "向右滑动（返回手势 / 桌面上一页）",
                "direction": "right",
                "node": -1,
                "role": "swipe",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "double_tap",
                "kind": "double_tap",
                "label": "双击屏幕中心",
                "node": -1,
                "role": "gesture",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "scroll_down",
                "kind": "scroll",
                "label": "向下滚动",
                "delta": 600,
                "node": -1,
                "role": "scroll",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "scroll_up",
                "kind": "scroll",
                "label": "向上滚动",
                "delta": -600,
                "node": -1,
                "role": "scroll",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "wait",
                "kind": "wait",
                "label": "等待界面刷新",
                "node": -1,
                "role": "wait",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                "id": "wait_long",
                "kind": "wait",
                "label": "等待 2 秒（动画/启动）",
                "node": -1,
                "role": "wait",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
            {
                # Five-second pause for app launch + splash / ad load.
                # Two seconds (``wait_long``) and three seconds are both
                # sometimes too short for apps that show a full-screen
                # ad before the home screen (闲鱼 / 淘宝 / 高德地图 all
                # do this on cold start, and on a slow network the ad
                # video can take 4–5 s to finish); without a dedicated
                # longer wait Jev either taps on the ad creative or
                # picks the post-ad screen as the baseline observation.
                # The LLM planner is explicitly told to insert a
                # ``WAIT_LONGER`` step right after every ``LAUNCH_APP``
                # (see ``_PLAN_SYSTEM`` in ``llm.py``).
                "id": "wait_longer",
                "kind": "wait",
                "label": "等待 5 秒（启动广告 / 启动页）",
                "node": -1,
                "role": "wait",
                "value": "",
                "rect": {"x": 0, "y": 0, "w": 0, "h": 0},
            },
        ]
    )
    return actions


# How long each ``wait`` / ``wait_long`` / ``wait_longer`` action sleeps.
# Kept in one place so :meth:`AutoX.act` and tests share the same source
# of truth. New wait tiers must be added here AND in :data:`_synthetic_actions`
# above; the ``act`` dispatch below is keyed by ``action["id"]``.
_WAIT_DURATIONS = {
    "wait": 0.1,
    "wait_long": 2.0,
    "wait_longer": 5.0,
}


def _build_text(ui_tree: dict) -> str:
    parts, total = [], 0
    for raw in _walk(ui_tree):
        t = raw.get("t")
        if not t:
            continue
        if total + len(t) > 6000:
            break
        parts.append(t)
        total += len(t)
    return "\n".join(parts)


def _package_name(ui_tree: dict, fallback: str = "") -> str:
    """Pull ``packageName`` out of the tree (root or any child).

    The MCP format puts it on whichever node has a different package than
    its parent, plus the root; we just walk and pick the first one we see.
    """
    p = ui_tree.get("p") or ui_tree.get("package") or ui_tree.get("packageName")
    if p:
        return p
    for raw in _walk(ui_tree):
        candidate = raw.get("p") or raw.get("package")
        if candidate:
            return candidate
    return fallback


def _activity_name(ui_tree: dict, fallback: str = "Android screen") -> str:
    """Best-effort activity name. Older trees include it as ``activity``."""
    a = ui_tree.get("activity") or ui_tree.get("currentActivity")
    if a:
        return a
    return fallback


def _guard_for(action) -> list:
    """Per-action guard. Used by :meth:`AutoX.fresh` right before input."""
    return [
        action["node"],
        action["rect"],
        action["role"],
        action["value"],
        action["kind"],
    ]


def fingerprint(state: dict) -> str:
    """Hash the parts of the state that should change between screens."""
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# OCR fallback for a11y-blocked apps
# ---------------------------------------------------------------------------

# A UI tree is considered "sparse" (and worth running OCR on) when:
# - it produced fewer than ``MIN_REAL_ACTIONS`` actionable elements, OR
# - it produced no text content at all.
# Apps like WeChat report ``{a: "d"}`` on the root node only, so the
# table collapses to just scroll/wait. We want to OCR the screen in
# that case so the agent can still see tab labels, list items, etc.
MIN_REAL_ACTIONS = 3
MIN_TEXT_LEN = 32


def _ocr_rect(bounds: dict) -> dict:
    """Convert OCR ``{top, left, bottom, right}`` into ``{x, y, w, h}``.

    ``bounds`` defaults to a zero-sized rect when the OCR backend omits
    bounds; the agent will still see the label.
    """
    top = int(bounds.get("top", 0) or 0)
    left = int(bounds.get("left", 0) or 0)
    bottom = int(bounds.get("bottom", top) or top)
    right = int(bounds.get("right", left) or left)
    return {"x": left, "y": top, "w": max(1, right - left), "h": max(1, bottom - top)}


def _build_ocr_actions(ocr_entries: list[dict]) -> list:
    """Synthesize a click-action table from OCR text + bounds.

    Every non-empty entry becomes a click action targeting the centre
    of the recognized text. We expose them with id ``ocrN`` so callers
    (and the test suite) can tell UI-tree-derived actions from
    OCR-derived ones.
    """
    actions: list = []
    counter = 0
    for entry in ocr_entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        bounds = entry.get("bounds") or {}
        rect = _ocr_rect(bounds)
        counter += 1
        node_id = f"ocr-{counter}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"
        actions.append(
            {
                "id": f"ocr{counter}",
                "kind": "click",
                "role": "text",
                "label": text,
                "value": text,
                "node": node_id,
                "rect": rect,
                "ocr": True,
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
            }
        )
    return actions


def _is_sparse_ui(page: dict) -> bool:
    """True when the observed UI tree alone is too thin to act on.

    Apps that disable accessibility (WeChat, some games) collapse the
    compact tree to a single root node; the policy would see only
    ``scroll_down / scroll_up / wait`` and never make progress. The
    OCR fallback exists for exactly that situation.
    """
    real = [a for a in page.get("actions", []) if a["id"].startswith("e") or a["id"].startswith("ocr")]
    if len(real) >= MIN_REAL_ACTIONS:
        return False
    if len(page.get("text", "").strip()) >= MIN_TEXT_LEN:
        return False
    return True


# ---------------------------------------------------------------------------
# Real device client
# ---------------------------------------------------------------------------

class AutoX:
    """Drive an AutoX.js MCP server over HTTP. Mirrors :class:`jev_ultrafast.browser.Browser`.

    The constructor only opens a connection; no implicit navigation.
    ``url`` is kept for parity with the browser harness — AutoX.js
    doesn't have a "URL" concept, so it acts as a label / device
    selector instead.

    Set ``ocr_fallback=True`` to transparently fall back to on-device
    OCR (``mcp.ocr(source="screenshot")``) when the UI tree is too
    sparse to act on. This is what unblocks apps that disable
    accessibility services (WeChat, some games) — the policy still
    gets a meaningful element list, but the action ids are prefixed
    ``ocrN`` instead of ``eN`` so traces stay honest.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        mcp: MCPClient | None = None,
        ocr_fallback: bool = False,
        settle_s: float = 0.35,
    ):
        self.url = url
        self.mcp = mcp or MCPClient.from_env()
        self.screen_w, self.screen_h = self.mcp.display_size()
        self.ocr_fallback = ocr_fallback
        # Android screen transitions take a few hundred milliseconds. The
        # agent observes immediately after ``act``, so without a short
        # pause the post-action observation can still show the previous
        # screen and ``page_changed`` comes back ``False``. (The browser
        # harness gets this for free via its observe settle loop.)
        self.settle_s = settle_s
        # Installed apps are probed once at construction so the
        # LAUNCH_APP synthetic actions always reflect what's actually on
        # the device. The probe is best-effort — when the MCP build
        # doesn't ship ``list_apps`` (or the phone is unreachable), we
        # fall back to the curated :data:`_KNOWN_APPS` list.
        # Installed apps are probed in :meth:`_ensure_installed_apps`,
        # which populates both ``installed_apps`` and ``launch_apps`` in
        # one shot (the second is the LAUNCH_APP surface after applying
        # the optional ``LAUNCH_APP_ALLOWLIST`` filter). Declaring them
        # here as attributes documents the type/shape and keeps
        # :func:`hasattr` happy for ``self.launch_apps`` reads scattered
        # across the codebase; we must NOT reassign ``launch_apps``
        # below ``_ensure_installed_apps()`` because the call already
        # set it — re-assigning to ``[]`` here would silently drop every
        # installed app and leave the Jev payload showing only the
        # curated :data:`_KNOWN_APPS` list.
        self.installed_apps: list[dict] = []
        self.launch_apps: list[dict] = []
        self._installed_apps_fetched = False
        self._ensure_installed_apps()
        # Cache the latest observation's elements so callers can ask
        # "what would Jev see right now" without re-running observe.
        self._last_actions: list = []

    @staticmethod
    def _filter_launch_apps(installed: list[dict]) -> list[dict]:
        """Apply ``LAUNCH_APP_ALLOWLIST`` env filtering.

        The env var is a comma-separated package list. Entries prefixed
        with ``-`` are removed (after the allowlist is applied). An
        empty string means "no filtering", so the default behaviour
        (no env var) keeps every installed app.
        """
        raw = (os.environ.get("LAUNCH_APP_ALLOWLIST") or "").strip()
        if not raw or not installed:
            return installed
        allow: set[str] = set()
        deny: set[str] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("-"):
                deny.add(token[1:])
            else:
                allow.add(token)
        out = []
        for row in installed:
            pkg = row.get("package") or ""
            if allow and pkg not in allow:
                continue
            if pkg in deny:
                continue
            out.append(row)
        return out or installed

    def list_installed_apps(self) -> list[dict]:
        """Public accessor used by :func:`mobile_jev_ultrafast.llm.plan_task`.

        Probes the device on first call; caches for the lifetime of the
        ``AutoX`` instance. Returns a copy so callers can't mutate the
        cache.
        """
        self._ensure_installed_apps()
        return list(self.installed_apps or [])

    def _ensure_installed_apps(self) -> None:
        """Probe the MCP for installed apps if we haven't already.

        Wrapped in a one-shot guard so the round-trip only happens once
        per ``AutoX`` lifetime. Failures degrade to ``[]`` and the
        LAUNCH_APP surface falls back to :data:`_KNOWN_APPS`.
        """
        if self._installed_apps_fetched:
            return
        self._installed_apps_fetched = True
        try:
            self.installed_apps = self.mcp.installed_apps() or []
        except Exception as exc:  # noqa: BLE001
            log = __import__("logging").getLogger(__name__)
            log.debug("installed_apps probe failed: %s", exc)
            self.installed_apps = []
        self.launch_apps = self._filter_launch_apps(self.installed_apps or [])

    # --- observation ------------------------------------------------------

    def _build_state(self, ui_tree: dict, screenshot: str | None = None) -> dict:
        # Make sure we have the LAUNCH_APP list before we build the
        # action table — the very first ``observe()`` would otherwise
        # surface only the curated :data:`_KNOWN_APPS` entries.
        self._ensure_installed_apps()
        actions = _build_actions(ui_tree, launch_apps=self.launch_apps)
        text = _build_text(ui_tree)
        w, h = self.screen_w, self.screen_h
        marker = [
            _package_name(ui_tree),
            _activity_name(ui_tree),
            text,
            [(a["node"], a["kind"], a["label"]) for a in actions],
        ]
        page_key = [
            _package_name(ui_tree),
            _activity_name(ui_tree),
            [(a["node"], a["value"]) for a in actions],
        ]
        guards = {a["node"]: _guard_for(a) for a in actions if a["kind"] != "wait"}
        state = {
            "url": f"{_package_name(ui_tree)}/{_activity_name(ui_tree)}",
            "title": _activity_name(ui_tree),
            "text": text,
            "scroll": {"y": 0, "height": h},
            "actions": actions,
            "marker": marker,
            "page_key": page_key,
            "guards": guards,
            "w": w,
            "h": h,
            "package": _package_name(ui_tree),
            "activity": _activity_name(ui_tree),
            "screenshot": screenshot or _BLANK_SCREENSHOT,
        }
        state["fingerprint"] = fingerprint(state)
        return state

    def observe(self, screenshot: bool = False):
        ui_tree = self.mcp.get_ui_tree()
        ss = self.mcp.screenshot(as_base64=True) if screenshot else None
        state = self._build_state(ui_tree, ss)
        if self.ocr_fallback and _is_sparse_ui(state):
            ocr_state = self._observe_via_ocr(ui_tree, ss, screenshot)
            return ocr_state
        return state

    def _observe_via_ocr(self, ui_tree: dict, ss: str | None, screenshot: bool) -> dict:
        """Build a state from a screenshot + OCR when the UI tree is empty.

        We keep the package / activity from the (sparse) UI tree so
        :attr:`state["url"]` is still informative, and reuse the
        existing screenshot if the caller already paid for one. The
        server-side ``ocr`` tool materialises its own screenshot via
        ``source="screenshot"`` when no inline base64 was supplied, so
        we don't force one here.
        """
        ocr_entries = self.mcp.ocr(source="screenshot")
        ocr_actions = _build_ocr_actions(ocr_entries)
        ocr_text = "\n".join((e.get("text") or "").strip() for e in ocr_entries if e.get("text"))
        package = _package_name(ui_tree)
        activity = _activity_name(ui_tree)
        w, h = self.screen_w, self.screen_h
        self._ensure_installed_apps()
        actions = list(ocr_actions) + _synthetic_actions(launch_apps=self.launch_apps)
        state = {
            "url": f"{package}/{activity}",
            "title": activity,
            "text": ocr_text,
            "scroll": {"y": 0, "height": h},
            "actions": actions,
            "w": w,
            "h": h,
            "package": package,
            "activity": activity,
            "screenshot": ss or _BLANK_SCREENSHOT,
            "marker": [
                package,
                activity,
                ocr_text,
                [(a["node"], a["kind"], a["label"]) for a in actions],
            ],
            "page_key": [
                package,
                activity,
                [(a["node"], a.get("value", "")) for a in actions],
            ],
            "guards": {a["node"]: _guard_for(a) for a in actions if a["kind"] != "wait"},
            "fallback": "ocr",
            "ocr_count": len(ocr_entries),
            "ocr_confidence_avg": (
                sum(e.get("confidence", 0.0) for e in ocr_entries) / len(ocr_entries)
                if ocr_entries
                else 0.0
            ),
        }
        state["fingerprint"] = fingerprint(state)
        return state

    # --- freshness -------------------------------------------------------

    def fresh(self, page, action=None) -> bool:
        try:
            current = self.observe(screenshot=False)
        except Exception:
            return False
        # Gestures (scroll / swipe / key) only depend on the screen being
        # the same kind of activity — an ad banner or toast flipping
        # between observe() calls must NOT invalidate a swipe, otherwise
        # scroll_down / scroll_up get caught by ``StalePage`` on app
        # home pages whose banner ad rotates every few seconds. Tighten
        # the check for clicks / fills so a stale element is still
        # detected (those need the action label to be present).
        if action is not None and action.get("kind") in {"scroll", "swipe", "key", "wait", "double_tap", "launch"}:
            return (
                current.get("package") == page.get("package")
                and current.get("activity") == page.get("activity")
            )
        # Whole-screen semantic match. jev-ultrafast compares a list of
        # (URL, scroll, viewport, safe-form-values, semantics). Android
        # has no equivalent of "safe form values" so the marker covers
        # label, role, value for every observed action.
        return current["marker"] == page["marker"]

    # --- execution -------------------------------------------------------

    def act(self, action, page, text: str | None = None):
        if not self.fresh(page, action):
            raise StalePage("Screen changed before this action executed.")
        kind = action["kind"]
        rect = action["rect"]
        cx = rect["x"] + rect["w"] // 2
        cy = rect["y"] + rect["h"] // 2
        if kind == "wait":
            # Tiered wait: 0.1s for trivial UI refresh, 2s for animations
            # / startup, 5s for the cold-launch splash + ad that
            # ``LAUNCH_APP`` triggers. Durations are looked up in
            # :data:`_WAIT_DURATIONS` so a new tier only needs to be
            # added there (and to :data:`_synthetic_actions`).
            time.sleep(_WAIT_DURATIONS.get(action["id"], 0.1))
            return {"executed": action["id"]}
        if kind == "scroll":
            # jev-ultrafast's ``scroll_up`` / ``scroll_down`` are named
            # after the *content* direction (deltaY<0 == wheel-up ==
            # viewport moves up the page). On a touchscreen the same
            # content-direction map would require a *opposite* finger
            # gesture, which clashes with the standard Chinese UX
            # vocabulary users actually use: ``"上滑"`` describes the
            # **finger** direction (手指从下往上滑), and Jev maps that
            # onto ``SCROLL_UP`` (``"向上滚动"``).
            #
            # We resolve the clash by treating the action name as the
            # *gesture* direction: ``scroll_up`` is a finger drag from
            # the bottom of the screen to the top, and ``scroll_down``
            # is the mirror. This matches every Chinese-speaking user's
            # intuition and what every other Android automation tool
            # does. (The previous code did the opposite and triggered
            # pull-to-refresh on Taobao whenever someone said "手机上滑".)
            x_mid = self.screen_w // 2
            # Keep the stroke inside [0.20h, 0.80h] so we never start in
            # the status bar (top ~0.08h) or end in the gesture-nav strip
            # (bottom ~0.08h); both can swallow the gesture.
            y_top = int(self.screen_h * 0.20)
            y_bot = int(self.screen_h * 0.80)
            if action["delta"] > 0:
                # ``scroll_down`` (delta=+600): finger drags downward,
                # top→bottom.
                y1, y2 = y_top, y_bot
            else:
                # ``scroll_up`` (delta=-600): finger drags upward,
                # bottom→top.
                y1, y2 = y_bot, y_top
            self.mcp.swipe(x_mid, y1, x_mid, y2, duration=_swipe_duration_ms())
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        if kind == "swipe":
            # Horizontal swipes are used to flip between launcher pages
            # and to dismiss bottom sheets via edge-gesture.
            direction = action.get("direction") or ""
            if direction == "left":
                x1, x2 = int(self.screen_w * 0.85), int(self.screen_w * 0.15)
            elif direction == "right":
                x1, x2 = int(self.screen_w * 0.15), int(self.screen_w * 0.85)
            else:
                x1, x2 = int(self.screen_w * 0.5), int(self.screen_w * 0.5)
            y_mid = self.screen_h // 2
            self.mcp.swipe(x1, y_mid, x2, y_mid, duration=_swipe_duration_ms())
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        if kind == "key":
            key = action.get("key") or ""
            if key == "home":
                self.mcp.press_home()
            elif key == "back":
                self.mcp.press_back()
            elif key == "recents":
                self.mcp.press_recents()
            else:
                raise ValueError(f"Unknown key action {key!r}")
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        if kind == "launch":
            pkg = action.get("value") or action.get("package") or ""
            if not pkg:
                raise ValueError("LAUNCH_APP action requires a package name")
            self.mcp.app_control(action="launch", package=pkg)
            # App launches need a longer settle than taps — the system
            # shows the launch animation, then the new app draws its
            # first frame. ``settle_s`` is rarely enough.
            time.sleep(max(self.settle_s, 1.0))
            return {"executed": action["id"]}
        if kind == "double_tap":
            self.mcp.tap(cx, cy)
            time.sleep(0.05)
            self.mcp.tap(cx, cy)
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        if kind == "click":
            self.mcp.tap(cx, cy)
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        if kind == "fill":
            self.mcp.tap(cx, cy)
            time.sleep(self.settle_s)
            if text:
                # ``setText`` is the canonical path. If the build lacks it,
                # we surface the error so the user can fix the MCP build;
                # inventing a paste fallback would silently type the wrong
                # text. The dashboard surfaces the error cleanly.
                self.mcp.set_text(text)
            return {"executed": action["id"]}
        if kind == "select":
            # Open the dropdown / spinner; the next observation will
            # surface the options as new actions. The policy is expected
            # to CLICK the desired one on the next loop.
            self.mcp.tap(cx, cy)
            time.sleep(self.settle_s)
            return {"executed": action["id"]}
        raise ValueError(f"Unknown action kind {kind!r}")

    def close(self):
        # MCP has no explicit "close tab"; drop the client.
        pass


# ---------------------------------------------------------------------------
# In-memory mock for local development and tests
# ---------------------------------------------------------------------------

@dataclass
class _MockAction:
    role: str
    label: str
    value: str = ""
    rect: tuple = (60, 200, 1020, 280)
    on_click: str | None = None
    checked: bool | None = None


# Canonical names. ``sound`` is accepted as an alias of ``ring`` so older
# callers keep working.
_RING_MODES = ("silent", "vibrate", "ring")
_RING_ALIASES = {
    "silent": "silent", "vibrate": "vibrate",
    "ring": "ring", "sound": "ring", "normal": "ring",
}

# The mock mirrors the reference device (HUAWEI LYA-AL00, Chinese
# locale), so its labels are Chinese. ``FakeAutoX`` is meant to stand in
# for the real phone in offline demos — English labels would let a goal
# pass here and fail on the device.
_RING_DISPLAY = {"silent": "静音", "vibrate": "振动", "ring": "响铃"}
_RING_LABEL_TO_MODE = {
    "silent": "silent", "静音": "silent",
    "vibrate": "vibrate", "振动": "vibrate",
    "ring": "ring", "sound": "ring", "响铃": "ring", "normal": "ring",
}


def _ring_mode_screen(mode: str) -> dict:
    """Build the sound-screen mock for one of three ring modes."""
    mode = _RING_ALIASES.get(mode, "silent")
    actions = []
    for index, candidate in enumerate(_RING_MODES):
        checked = candidate == mode
        actions.append(
            _MockAction(
                role="radio",
                label=_RING_DISPLAY[candidate],
                value="true" if checked else "false",
                rect=(80, 400 + index * 200, 1000, 580 + index * 200),
                checked=checked,
            )
        )
    label_text = ", ".join(
        f"{_RING_DISPLAY[m]}（已选中）" if m == mode else _RING_DISPLAY[m]
        for m in _RING_MODES
    )
    return {
        "activity": "com.example.app.SoundActivity",
        "title": "Demo · Sound",
        "text": f"声音和振动\n声音模式\n{label_text}",
        "actions": actions,
    }


_MOCK_SCREENS = {
    "home": {
        "activity": "com.example.app.HomeActivity",
        "title": "Demo · Home",
        "text": (
            "Demo Home\n"
            "Search the catalog\n"
            "Open Settings\n"
            "About this demo"
        ),
        "actions": [
            _MockAction("textbox", "Search products", ""),
            _MockAction("button", "Search", ""),
            _MockAction("button", "Open Settings", "", on_click="settings"),
        ],
    },
    "settings": {
        "activity": "com.example.app.SettingsActivity",
        "title": "Demo · Settings",
        "text": "Settings\nNotifications\nSound\nAccount\nAbout",
        "actions": [
            _MockAction("switch", "Notifications", "true"),
            _MockAction("button", "Sound", "", on_click="sound"),
            _MockAction("button", "Account", "", on_click="account"),
            _MockAction("button", "About", "", on_click="about"),
        ],
    },
    "sound": _ring_mode_screen("silent"),
    "account": {
        "activity": "com.example.app.AccountActivity",
        "title": "Demo · Account",
        "text": "Account\nSigned in as demo@example.com",
        "actions": [_MockAction("button", "Sign out", "")],
    },
    "about": {
        "activity": "com.example.app.AboutActivity",
        "title": "Demo · About",
        "text": "About\nmobile-jev-ultrafast 0.1.0\nAndroid version: 14",
        "actions": [],
    },
}


def _mock_action_dict(m: _MockAction):
    kind = "fill" if m.role == "textbox" else "click"
    action = {
        "id": "",  # filled in by FakeAutoX.observe
        "kind": kind,
        "role": m.role,
        "label": m.label,
        "value": m.value,
        "node": f"mock-{m.label}",
        "rect": {"x": m.rect[0], "y": m.rect[1], "w": m.rect[2] - m.rect[0], "h": m.rect[3] - m.rect[1]},
    }
    if m.checked is not None:
        action["checked"] = "true" if m.checked else "false"
    return action


class FakeAutoX:
    """Returns a hand-crafted mock so the inspector & tests run locally."""

    def __init__(self, url=None, initial_ring: str = "silent"):
        self.url = url
        self._state_name = "home"
        # Mirror the device's "current ringer mode" so the toggle demo
        # can observe a non-trivial cycle (silent → vibrate → sound → ...).
        self._ring_mode = _RING_ALIASES.get(initial_ring, "silent")

    @property
    def _state(self):
        if self._state_name == "sound":
            return _ring_mode_screen(self._ring_mode)
        return _MOCK_SCREENS[self._state_name]

    def observe(self, screenshot: bool = False) -> dict:
        m = self._state
        actions = [_mock_action_dict(a) for a in m["actions"]]
        for i, a in enumerate(actions, start=1):
            a["id"] = f"e{i}"
        actions.extend(_synthetic_actions(launch_apps=None))
        page = {
            "url": f"com.example.app/{m['activity']}",
            "title": m["title"],
            "text": m["text"],
            "scroll": {"y": 0, "height": 2400},
            "actions": actions,
            "w": 1080,
            "h": 2400,
            "package": "com.example.app",
            "activity": m["activity"],
            "screenshot": _BLANK_SCREENSHOT,
        }
        page["marker"] = [
            page["package"],
            page["activity"],
            page["text"],
            [(a["node"], a["kind"], a["label"]) for a in actions],
        ]
        page["page_key"] = [
            page["package"],
            page["activity"],
            [(a["node"], a["value"]) for a in actions],
        ]
        page["guards"] = {a["node"]: _guard_for(a) for a in actions if a["kind"] != "wait"}
        page["fingerprint"] = fingerprint(page)
        return page

    def fresh(self, page, action=None) -> bool:
        # Mock never gets stale, mirroring test fixtures.
        return True

    def act(self, action, page, text=None):
        kind = action["kind"]
        if kind == "scroll":
            return {"executed": action["id"]}
        if kind == "swipe":
            return {"executed": action["id"]}
        if kind == "key":
            # ``press_home`` returns the mock to its home screen so the
            # next observe() looks like a fresh launcher; back/recents
            # are no-ops since the mock has no stack.
            if action.get("key") == "home":
                self._state_name = "home"
            return {"executed": action["id"]}
        if kind == "launch":
            pkg = action.get("value") or ""
            # Map a couple of well-known packages to mock screens so the
            # loop is testable without a phone.
            route = {
                "com.android.settings": "settings",
                "com.example.app": "home",
            }.get(pkg)
            if route:
                self._state_name = route
            return {"executed": action["id"]}
        if kind == "double_tap":
            return {"executed": action["id"]}
        # Sound screen: tapping any of the three radios updates the
        # tracked ring mode so the next ``observe()`` reflects it.
        # This includes taps on the clickable *row* (whose inherited
        # label is the mode name) as well as on the radio itself.
        if self._state_name == "sound":
            mode = _RING_LABEL_TO_MODE.get((action.get("label") or "").strip().lower())
            if mode:
                self._ring_mode = mode
                return {"executed": action["id"]}
        for tmpl in self._state["actions"]:
            if tmpl.label == action["label"] and tmpl.on_click:
                self._state_name = tmpl.on_click
                return {"executed": action["id"]}
        if kind == "fill":
            for tmpl in self._state["actions"]:
                if tmpl.label == action["label"] and tmpl.role == "textbox":
                    tmpl.value = text or ""
                    return {"executed": action["id"]}
        return {"executed": action["id"]}

    def list_installed_apps(self) -> list[dict]:
        return [
            {"label": "设置", "package": "com.android.settings", "system": True},
            {"label": "Demo", "package": "com.example.app", "system": False},
        ]

    def close(self):
        pass
