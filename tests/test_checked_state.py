"""Tests for checked / selected state propagation in ``_build_actions``.

Android renders a radio row as a clickable container holding a label
``TextView`` and a state-bearing ``RadioButton``. The state lives on the
latter, but the *label* the policy reasons about is on the former, so
``_build_actions`` propagates ``checked`` across the row.

The synthetic tree below mirrors the real HUAWEI ``Settings → 声音和振动``
screen dumped from the device (HUAWEI LYA-AL00, Android 10). It assumes
the MCP server has been patched to emit ``checked`` — see
``docs/REQUIREMENTS_MY_AUTOX.md``. Before that patch lands the tree has
no ``checked`` field and the actions simply lack the attribute (which is
the current, already-working behaviour).
"""

from __future__ import annotations

from my_autox_server.autox import (
    _build_actions,
    _classify,
    _interaction_flags,
    _norm_checked,
    _sibling_checked,
)

# --- the real device structure, with the ``checked`` field filled in -------

def _sound_mode_row(label: str, *, checked: bool) -> dict:
    return {
        "c": "LinearLayout",
        "b": [0, 0, 0, 0],
        "children": [
            {
                "c": "LinearLayout",
                "a": "cf",
                "id": "sound_mode_click",
                "b": [72, 366, 1008, 546],
                "children": [
                    {"c": "ImageView", "id": "sound_mode_icon", "b": [90, 380, 170, 460]},
                    {
                        "c": "TextView",
                        "id": "sound_mode_title",
                        "t": label,
                        "b": [200, 380, 700, 460],
                    },
                    {
                        "c": "RadioButton",
                        "id": "sound_mode_radiobutton",
                        "a": "k",
                        "checked": checked,
                        "b": [930, 390, 1000, 460],
                    },
                ],
            }
        ],
    }


SOUND_TREE = {
    "p": "com.android.settings",
    "activity": "MiuiSoundSettings",
    "c": "FrameLayout",
    "b": [0, 0, 1080, 2400],
    "a": "",
    "children": [
        {
            "c": "RecyclerView",
            "a": "fs",
            "id": "list",
            "b": [0, 200, 1080, 2200],
            "children": [
                {
                    "c": "LinearLayout",
                    "b": [0, 260, 1080, 300],
                    "children": [
                        {"c": "TextView", "t": "声音模式", "id": "title", "b": [72, 260, 240, 300]}
                    ],
                },
                _sound_mode_row("响铃", checked=False),
                _sound_mode_row("振动", checked=True),
                _sound_mode_row("静音", checked=False),
            ],
        }
    ],
}


def _by_label(actions, label):
    return next(a for a in actions if a["label"] == label)


# --- unit helpers ----------------------------------------------------------

def test_norm_checked_handles_json_shapes():
    assert _norm_checked(True) == "true"
    assert _norm_checked(False) == "false"
    assert _norm_checked("true") == "true"
    assert _norm_checked("1") == "true"
    assert _norm_checked("on") == "true"
    assert _norm_checked("false") == "false"
    assert _norm_checked("0") == "false"
    assert _norm_checked(None) is None
    assert _norm_checked("maybe") is None


def test_interaction_flags_recognise_checkable_and_selected():
    assert "k" in _interaction_flags({"a": "k"})
    assert "k" in _interaction_flags({"a": "ck"})
    assert "x" in _interaction_flags({"a": "x"})
    assert "k" in _interaction_flags({"a": "checkable"})


def test_classify_promotes_checkable_layout_to_button():
    assert _classify({"c": "LinearLayout", "a": "cf"}) == "button"
    assert _classify({"c": "LinearLayout", "a": "k"}) == "button"
    assert _classify({"c": "RadioButton", "a": "k"}) == "radio"


def test_sibling_checked_reads_label_sibling_state():
    row = _sound_mode_row("振动", checked=True)["children"][0]
    label_node = next(c for c in row["children"] if c.get("t") == "振动")
    assert _sibling_checked(label_node, row) == "true"


def test_sibling_checked_reads_container_descendant_state():
    row = _sound_mode_row("振动", checked=True)["children"][0]
    # The clickable container has no text, so it inherits from its
    # RadioButton child.
    assert _sibling_checked(row, None) == "true"


# --- integration: the element table ---------------------------------------

def test_checked_state_propagates_to_every_visible_row_element():
    actions = _build_actions(SOUND_TREE)
    labels = {a["label"] for a in actions}
    assert {"响铃", "振动", "静音"} <= labels

    assert _by_label(actions, "响铃")["checked"] == "false"
    assert _by_label(actions, "振动")["checked"] == "true"
    assert _by_label(actions, "静音")["checked"] == "false"


def test_clickable_container_inherits_label_and_state():
    """The clickable row itself must carry both label and state.

    Before the client-side association the row surfaced as an anonymous
    ``linearlayout`` action; the policy had no way to tie it to a mode.
    """
    actions = _build_actions(SOUND_TREE)
    row_actions = [a for a in actions if a["role"] == "button" and a.get("checked") is not None]
    by_label = {a["label"]: a for a in row_actions}
    assert by_label["振动"]["checked"] == "true"
    assert by_label["响铃"]["checked"] == "false"
    # The container takes the label of its first labelled descendant.
    assert by_label["振动"]["rect"]["w"] > 0


def test_exactly_one_row_is_checked():
    actions = _build_actions(SOUND_TREE)
    radio_rows = [
        a for a in actions
        if a["label"] in {"响铃", "振动", "静音"} and a["role"] == "radio"
    ]
    assert len(radio_rows) == 3
    assert sum(1 for a in radio_rows if a["checked"] == "true") == 1


def test_no_checked_field_when_server_has_not_been_patched():
    """An unpatched server emits no ``checked`` — actions must not invent one."""
    import copy

    tree = copy.deepcopy(SOUND_TREE)

    def strip(node):
        node.pop("checked", None)
        for child in node.get("children") or []:
            strip(child)

    strip(tree)
    actions = _build_actions(tree)
    assert all("checked" not in a for a in actions)


def test_selected_and_disabled_are_surfaced():
    tree = {
        "p": "com.example",
        "c": "FrameLayout",
        "b": [0, 0, 100, 100],
        "a": "",
        "children": [
            {
                "c": "TextView",
                "t": "Highlighted row",
                "a": "x",
                "selected": True,
                "b": [0, 0, 100, 40],
            },
            {
                "c": "Button",
                "t": "Greyed out",
                "a": "cd",
                "enabled": False,
                "b": [0, 40, 100, 80],
            },
        ],
    }
    actions = _build_actions(tree)
    highlighted = _by_label(actions, "Highlighted row")
    assert highlighted["selected"] == "true"
    disabled = _by_label(actions, "Greyed out")
    assert disabled["enabled"] == "false"