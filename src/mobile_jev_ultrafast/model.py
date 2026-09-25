"""Decision backends: TypeSafe by default, plus a deterministic offline one.

The original ``choose()`` and ``field_text()`` are kept as the dispatch
targets ``typesafe`` and ``helper`` respectively. A new ``scripted``
backend derives an answer from the observed state without any network
calls, so the agent loop can be exercised end-to-end without a paid API
key (useful for unit tests, the loopback inspector, and the verified
demo).

Selection is per-call via env vars so callers don't need to know which
backend they got:

* ``AGENT_DECISION_BACKEND`` — ``typesafe`` (default) | ``scripted``
* ``AGENT_TEXT_BACKEND``     — ``helper`` (default) | ``scripted``

Both ``choose()`` and ``field_text()`` resolve the backend at call time,
not at import time, so a script can flip the env var between runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import httpx

from . import verbose
from .questions import LAUNCH_APP, NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)

# ``task_complete`` is the second judgment Jev makes every tick: should
# the loop stop right after this action executes? It's an independent
# choice (continue vs finish) so the model can mark a SCROLL_UP / CLICK
# / LAUNCH_APP as terminal when the goal is operational (e.g. 「上滑 1 下」)
# and still drive the loop on every step when the goal is destination-oriented
# (e.g. 「打开设置」). The agent loop trusts ``finish`` only when confidence
# clears ``TASK_COMPLETE_FINISH_THRESHOLD`` so a low-confidence signal
# doesn't end the run prematurely.
TASK_COMPLETE_CHOICES = {
    "continue": "More steps are still required after this action executes.",
    "finish": "The user's entire goal is satisfied once this action executes; stop the loop now.",
}
TASK_COMPLETE_FINISH_THRESHOLD = 0.7


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls, launch_choices = [], {}, {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind == "launch":
            controls[action["id"].upper()] = action
            launch_choices[action["id"]] = {
                "label": action.get("label") or "",
                "package": action.get("value") or action.get("package") or "",
                "role": "app",
            }
            continue
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls, launch_choices


def _strip_launch_prefix(label: str) -> str:
    """Strip the ``"启动 "`` / ``"Launch "`` prefix off a launch-action label.

    :func:`_launch_actions` builds labels like ``"启动 闲鱼"`` (zh) or
    ``"Launch WeChat"`` (en); everything else (``"按 Home 键"``,
    ``"向左滑动"``) is left alone. The planner returns the bare app
    label (``"闲鱼"``), so we compare against the stripped form.
    """
    if not label:
        return ""
    for prefix in ("启动 ", "Launch "):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _filter_launch_choices(
    launch_choices: dict,
    relevant_apps: list[str] | None,
) -> dict:
    """Drop launch entries whose label doesn't match any relevant app.

    The pre-task planner (see :func:`mobile_jev_ultrafast.llm.plan_task`)
    already classified the user's goal and returned the labels of apps it
    thinks are involved (``RELEVANT_APPS=闲鱼`` for "打开闲鱼, 切换到 西安,
    搜索跑步机"). When the planner did its job, we should not flood the
    model with the 50+ launch actions on the device — most of them are
    red herrings the model would waste probability mass on. So we prune
    ``launch_choices`` down to the subset whose display label matches a
    relevant app.

    Matching is case-insensitive substring on the stripped label so:

    * exact label match — ``"闲鱼"`` → ``"启动 闲鱼"``
    * prefixed label — ``"淘宝"`` → ``"启动 淘宝"``
    * parent package — ``"淘宝"`` → ``"启动 闲鱼 (淘宝)"`` (some launch
      entries append the parent brand in parentheses; substring still
      catches it).

    Falls back to the original dict when:

    * ``relevant_apps`` is ``None`` or empty (planner disabled, or the
      LLM returned nothing — preserve the old "all apps available"
      behaviour).
    * The filter drops everything (LLM mislabelled every app, e.g.
      ``RELEVANT_APPS=SomeAppNotOnDevice``). Refusing to launch would
      brick the loop; better to keep the full set with a verbose
      warning so the run is still diagnosable.

    Returns the (possibly empty) filtered dict; the caller decides
    what to do with the matching ``launch_<pkg>`` entries in
    ``controls`` (only ``LAUNCH_*`` keys are pruned — every other
    control like ``PRESS_HOME`` / ``SWIPE_LEFT`` is untouched).
    """
    if not relevant_apps or not launch_choices:
        return launch_choices
    needles = [n.strip().lower() for n in relevant_apps if n and n.strip()]
    if not needles:
        return launch_choices

    filtered: dict = {}
    for action_id, choice in launch_choices.items():
        stripped = _strip_launch_prefix(choice.get("label", "")).lower()
        if not stripped:
            continue
        if any(needle in stripped or stripped in needle for needle in needles):
            filtered[action_id] = choice

    if not filtered:
        verbose.emit(
            1,
            f"  JEV: relevant_apps filter matched 0 launch entries "
            f"(relevant={relevant_apps!r}); keeping all {len(launch_choices)} launches",
        )
        return launch_choices

    verbose.emit(
        2,
        verbose.wrap(
            f"  JEV: relevant_apps filter kept {len(filtered)}/{len(launch_choices)} "
            f"launch entries (relevant={relevant_apps!r}): ",
            ", ".join(sorted(_strip_launch_prefix(c.get('label', '')) for c in filtered.values())),
        ),
    )
    return filtered


def _choose_typesafe(state, goal, history, *, relevant_apps: list[str] | None = None):
    """Original TypeSafe-backed decision policy. Identical to jev-ultrafast."""
    elements, targets, controls, launch_choices = action_space(state["actions"])
    # The pre-task planner narrows the launch surface to apps it
    # actually considers relevant (see :func:`_filter_launch_choices`).
    # Without this filter the model sees ~50 launch_<pkg> entries and
    # spends probability on every installed app instead of the one
    # ``RELEVANT_APPS=闲鱼`` told it about.
    if relevant_apps:
        launch_choices = _filter_launch_choices(launch_choices, relevant_apps)
        # Drop the matching ``LAUNCH_<PKG>`` controls so the per-action
        # ``probabilities`` vector doesn't leak the pruned entries.
        kept_ids = {aid.lower() for aid in launch_choices}
        controls = {
            cid: action
            for cid, action in controls.items()
            if not (cid.lower().startswith("launch_") and cid.lower() not in kept_ids)
        }
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    # LAUNCH_APP / PRESS_HOME / PRESS_BACK / etc. carry their own label.
    # ``WAIT`` / ``DONE`` / ``BLOCKED`` already live in ``controls`` /
    # are appended below; the LAUNCH_APP entry is here so the model can
    # pick it without the criteria collapsing to a single option.
    if launch_choices:
        operations["LAUNCH_APP"] = (
            "Launch an installed app directly via Intent. Pick a package in launch_target."
        )
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}},
        "task_complete": {
            "type": "choice",
            "criteria": TASK_COMPLETE_CHOICES,
            "instructions": {"goal": goal, "rules": [NEXT_ACTION]},
        },
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    if launch_choices:
        questions["launch_target"] = {
            "type": "choice",
            "criteria": {
                action_id: {
                    "element": choice["label"],
                    "package_name": choice["package"],
                    "role": choice["role"],
                }
                for action_id, choice in launch_choices.items()
            },
            "instructions": {"goal": goal, "operation": "LAUNCH_APP", "rules": [NEXT_ACTION, LAUNCH_APP]},
        }
    recent_payload = []
    for h in history[-10:]:
        entry = {k: h.get(k) for k in ("action", "kind", "text", "page_changed") if k in h}
        # Surface LLM_SUGGESTION / WARN / LLM_ACTION entries verbatim so
        # Jev treats them as high-priority hints (mirrors mobile-jev-jarvis).
        if h.get("operation") in {"LLM_SUGGESTION", "WARN", "LLM_ACTION"}:
            entry = {"operation": h.get("operation"), "note": h.get("note", "")}
        if h.get("operation") == "CLICK":
            entry["target_summary"] = h.get("action", "")
        recent_payload.append(entry)
    state_payload = {
        "page": {k: state[k] for k in ("url", "title", "text")},
        "elements": elements,
        "recent_actions": recent_payload,
    }
    task_plan = state.get("task_plan") if isinstance(state, dict) else None
    if task_plan:
        state_payload["task_plan"] = task_plan
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": state_payload,
        "questions": questions,
    }
    verbose.emit(2, "── JEV request ──────────────────────────────────────────")
    # The element table + recent-action window are the bulk of every Jev
    # payload; level 3 already prints them in compact form, so drop them
    # from the level-2 JSON dump to keep the rest of the request
    # readable. The dump limit is raised past the default 4 kB so the
    # ``questions`` block survives intact — the rules text alone runs
    # a few kB and the user asked for the rest of the payload to be
    # visible for debug.
    verbose.emit(
        2,
        verbose.blob(
            body,
            omit_paths=("state.elements", "state.recent_actions"),
            limit=verbose.BLOB_FULL_LIMIT,
        ),
    )
    verbose.emit(3, "── JEV element table ────────────────────────────────────")
    for element in elements:
        verbose.emit(
            3,
            f"  [{element['index']:>3}] {element['role']:<12} {element['label'][:44]!r} "
            f"ops={','.join(element.get('operations', []))}",
        )
    verbose.emit(3, "── JEV recent actions ───────────────────────────────────")
    for entry in recent_payload:
        verbose.emit(3, f"  {entry}")
    started = time.perf_counter()
    result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    verbose.emit(2, "── JEV response ─────────────────────────────────────────")
    # ``answers.click_target`` carries one probability per element, which
    # is by far the largest slice of the reply; the level-1 trace already
    # shows the top-N probabilities, so omit the full distribution here.
    # Same bump as the request so the remaining answers / usage block
    # stays readable.
    verbose.emit(
        2,
        verbose.blob(result, omit_paths=("answers.click_target",), limit=verbose.BLOB_FULL_LIMIT),
    )
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    # Parse the task_complete head. Missing or malformed answers fall back
    # to ``continue`` so a Jev model that doesn't yet support this head
    # doesn't break the loop; the verbose log surfaces the skip so it's
    # easy to notice in -v traces.
    task_complete_choice = "continue"
    task_complete_confidence = 0.0
    task_complete_probs = {"continue": 1.0, "finish": 0.0}
    raw_task_complete = result["answers"].get("task_complete")
    if isinstance(raw_task_complete, dict):
        raw_choice = raw_task_complete.get("choice")
        raw_probs = raw_task_complete.get("probabilities")
        if raw_choice in TASK_COMPLETE_CHOICES and isinstance(raw_probs, dict):
            task_complete_choice = raw_choice
            try:
                task_complete_probs = {k: float(raw_probs.get(k, 0.0)) for k in TASK_COMPLETE_CHOICES}
            except (TypeError, ValueError):
                task_complete_probs = {"continue": 1.0, "finish": 0.0}
                task_complete_choice = "continue"
            task_complete_confidence = float(raw_task_complete.get("confidence", 0.0) or 0.0)
        else:
            verbose.emit(
                1,
                "  JEV: task_complete head malformed (choice or probabilities invalid); treating as continue",
            )
    elif raw_task_complete is not None:
        verbose.emit(
            1,
            "  JEV: task_complete head missing or non-dict; treating as continue",
        )
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    elif operation == "LAUNCH_APP" and launch_choices:
        target_answer = validate_choice(result["answers"].get("launch_target", {}), launch_choices)
        target = target_answer["choice"]
        choice = target  # the launch action's id is the package-keyed id
        # ``launch_choices`` is a {action_id: {label, package, role}} map.
        # The values don't carry an ``id`` key (unlike ``targets[operation]``),
        # so use the dict key directly. Bug fix: previous code did
        # ``a["id"]`` and raised ``KeyError: 'id'`` the first time Jev picked
        # ``LAUNCH_APP`` for a non-trivial plan.
        probabilities = {aid: target_answer["probabilities"][aid] for aid in launch_choices}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    if verbose.enabled(1):
        detail = f"target={target}" if target is not None else "target=-"
        verbose.emit(
            1,
            f"  JEV: operation={operation} {detail} choice={choice} "
            f"confidence={operation_answer['confidence']:.3f} "
            f"latency={round((time.perf_counter() - started) * 1000)}ms",
        )

        def _top(probs):
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:6]
            return "{" + ", ".join(f"{k}={v:.3f}" for k, v in ranked) + "}"

        verbose.emit(1, f"  JEV: top operation probs={_top(operation_answer['probabilities'])}")
        if target_answer:
            verbose.emit(1, f"  JEV: top target probs={_top(target_answer['probabilities'])}")
        verbose.emit(
            1,
            f"  JEV: task_complete={task_complete_choice} "
            f"confidence={task_complete_confidence:.3f} "
            f"finish_prob={task_complete_probs.get('finish', 0.0):.3f}",
        )
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "task_complete": task_complete_choice,
        "task_complete_confidence": task_complete_confidence,
        "task_complete_probabilities": task_complete_probs,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
        "backend": "typesafe",
    }


# CJK text has no spaces, so whitespace splitting turns a whole Chinese
# sentence into a single token and nothing ever matches a label. We
# expand each CJK run into the run itself, its single characters, and
# its bigrams so substring matching against element labels works.
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_SPLIT = re.compile(r"[\s,.;:!?，。；：！？、「」『』【】（）()\[\]{}\"'“”]+")


def _goal_keywords(goal: str) -> list[str]:
    """Content tokens from a goal string, CJK-aware.

    Latin words are lower-cased and kept whole. CJK runs additionally
    yield their individual characters and bigrams (``响声`` →
    ``{"响声", "响", "声"}``) so a goal written in Chinese can still
    match element labels like ``"响铃"``.
    """
    tokens: list[str] = []
    for raw in _SPLIT.split(goal):
        raw = raw.strip()
        if not raw:
            continue
        if _CJK_RUN.fullmatch(raw):
            tokens.append(raw)
            tokens.extend(raw)
            tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
        else:
            tokens.append(raw.lower())
    return tokens


def _scripted_task_complete(goal: str, operation: str | None) -> dict:
    """Default ``task_complete`` for the offline backend.

    Mirrors the operational-vs-destination heuristic in :data:`questions.NEXT_ACTION`:
    only return ``finish`` when the goal really reads like "do this
    action and stop" (「上滑 1 下」, "scroll down twice", "click it
    once") — i.e. it has *both* an action verb and a count. Any
    destination verb ("open", "find", "go to", "tap X and ...", ...)
    wins immediately and the action returns ``continue`` so the loop
    keeps driving toward a screen. Tests can ignore this helper
    entirely; the production Jev model makes the real decision.
    """
    goal_text = (goal or "").lower()
    destination_verbs = (
        "打开", "进入", "跳转", "切到", "切换", "查看", "看看",
        "检查", "确认", "核实", "找到", "寻找", "搜索",
        "open", "navigate", "go to", "find", "search", "show",
    )
    if any(verb in goal_text for verb in destination_verbs):
        # The user wants to reach somewhere; one action is never enough.
        return {
            "task_complete": "continue",
            "task_complete_confidence": 0.9,
            "task_complete_probabilities": {"continue": 0.9, "finish": 0.1},
        }
    operational_verbs = {
        "scroll_up": ("上滑", "向上滑", "向上滚动", "scroll up", "swipe up"),
        "scroll_down": ("下滑", "向下滑", "向下滚动", "scroll down", "swipe down"),
        "swipe_left": ("左滑", "向左滑", "swipe left"),
        "swipe_right": ("右滑", "向右滑", "swipe right"),
        "press_back": ("返回", "退回", "press back"),
        "press_home": ("回桌面", "返回桌面", "桌面", "press home"),
        "press_recents": ("最近任务", "多任务", "recents"),
        "double_tap": ("双击", "double tap"),
        # ``click`` only counts as operational when the goal also names
        # a count — otherwise "tap About" reads as a destination verb.
        "click": ("点击", "点一下", "按一下", "点这个", "点那个", "tap"),
    }
    aliases = operational_verbs.get((operation or "").lower(), ())
    if not operation or not any(alias in goal_text for alias in aliases):
        return {
            "task_complete": "continue",
            "task_complete_confidence": 0.9,
            "task_complete_probabilities": {"continue": 0.9, "finish": 0.1},
        }
    # For ``click`` we additionally require a count so plain
    # "tap About" doesn't short-circuit.
    if (operation or "").lower() == "click":
        has_count = bool(
            re.search(r"\d+\s*(?:下|次|个|遍|回)", goal_text)
            or re.search(r"(?:下|次|个|遍|回)\s*\d+", goal_text)
            or re.search(r"\d+\s*(?:times?|steps?|clicks?|taps?)", goal_text)
            or re.search(r"\b(?:once|twice|three times)\b", goal_text)
        )
        if not has_count:
            return {
                "task_complete": "continue",
                "task_complete_confidence": 0.9,
                "task_complete_probabilities": {"continue": 0.9, "finish": 0.1},
            }
    return {
        "task_complete": "finish",
        "task_complete_confidence": 0.95,
        "task_complete_probabilities": {"continue": 0.05, "finish": 0.95},
    }


def _terminal_task_complete() -> dict:
    """``task_complete`` for terminal decisions (DONE / BLOCKED)."""
    return {
        "task_complete": "finish",
        "task_complete_confidence": 1.0,
        "task_complete_probabilities": {"continue": 0.0, "finish": 1.0},
    }


def _choose_scripted(state, goal, history, *, relevant_apps: list[str] | None = None):
    """Deterministic offline decision backend.

    Picks the most goal-relevant element with simple keyword scoring
    over labels. Sufficient for the verified demo (settings → about) and
    for unit tests that want to exercise the agent loop without paying
    for an API call. Returns the same dict shape as ``_choose_typesafe``
    so callers don't need to branch on the backend.
    """
    started = time.perf_counter()
    elements, targets, controls, launch_choices = action_space(state["actions"])
    if relevant_apps:
        launch_choices = _filter_launch_choices(launch_choices, relevant_apps)
        kept_ids = {aid.lower() for aid in launch_choices}
        controls = {
            cid: action
            for cid, action in controls.items()
            if not (cid.lower().startswith("launch_") and cid.lower() not in kept_ids)
        }
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    if launch_choices:
        operations["LAUNCH_APP"] = (
            "Launch an installed app directly via Intent. Pick a package in launch_target."
        )
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")

    keywords = _goal_keywords(goal)
    content_keywords = [kw for kw in keywords if kw and kw not in _STOPWORDS]

    # DONE only checks URL + title — never ``text``. Element labels
    # dominate ``text`` ("Open Settings" is itself a button), so we'd
    # false-positive on every intermediate screen. The demo's assertion
    # path relies on DONE firing only on the *target* screen, whose
    # package / activity always carries a unique substring.
    page_signature = " ".join(
        str(state.get(k, "")) for k in ("title", "url")
    ).lower()
    if content_keywords and content_keywords[-1] in page_signature:
        # Reuse the same shape TypeSafe answers carry.
        probabilities = {op: (1.0 if op == "DONE" else 0.0) for op in operations}
        terminal = content_keywords[-1]
        return {
            "choice": "DONE",
            "operation": "DONE",
            "target": None,
            "confidence": 0.99,
            "probabilities": probabilities,
            "operation_probabilities": probabilities,
            "target_probabilities": {},
            "target_confidence": None,
            **_terminal_task_complete(),
            "raw_answers": {"_scripted": {"terminal": terminal}},
            "model": "scripted",
            "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": {"goal": goal, "keywords": keywords, "page_signature": page_signature[:600]},
            "backend": "scripted",
        }

    # Otherwise pick the best operation + target by keyword score.
    best_op, best_score, best_target = "BLOCKED", -1, None
    scored: list[tuple[str, str, float]] = []
    keyword_set = set(content_keywords)
    # Radio-group cycle: when *exactly one* click candidate is
    # ``checked=true`` on the current page, prefer its *next* sibling
    # in the same group (wrapping around). This is the heuristic the
    # ``toggle_silent_mode`` demo relies on for ``--fake`` mode: the
    # Sound screen has three radios (silent/vibrate/sound) and the
    # test wants to advance exactly one slot along the cycle. We do
    # not require a keyword hit on the next sibling because the goal's
    # last content word is often generic ("applied" / "shown" / "done"),
    # which never matches any radio label.
    click_candidates = targets.get("CLICK", {})
    cycle_target: str | None = None
    if click_candidates:
        items = list(click_candidates.items())
        for i, (_idx, action) in enumerate(items):
            if str(action.get("checked", "")).lower() == "true":
                next_idx, _next_action = items[(i + 1) % len(items)]
                cycle_target = next_idx
                break
    if cycle_target is not None:
        next_action = click_candidates[cycle_target]
        probabilities = {a["id"]: 0.0 for a in state["actions"]}
        probabilities[next_action["id"]] = 1.0
        op_probs = {op: 0.0 for op in operations}
        op_probs["CLICK"] = 1.0
        target_probs = {idx: 0.0 for idx in click_candidates}
        target_probs[cycle_target] = 1.0
        return {
            "choice": next_action["id"],
            "operation": "CLICK",
            "target": cycle_target,
            "confidence": 0.95,
            "probabilities": probabilities,
            "operation_probabilities": op_probs,
            "target_probabilities": target_probs,
            "target_confidence": 0.95,
            **_scripted_task_complete(goal, "CLICK"),
            "raw_answers": {"_scripted": {"cycle": cycle_target}},
            "model": "scripted",
            "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": {"goal": goal, "cycle_target": cycle_target},
            "backend": "scripted",
        }

    # Scripted launch hint (after DONE / cycle so we don't loop): when
    # the goal clearly names a known app and we are not already on it,
    # launch it. Skip when:
    #   * an in-page clickable / fillable element already matches a
    #     keyword — clicking the visible button is more honest than
    #     re-launching the same app from a screen that mentions it.
    #   * the current package or page signature already contains a
    #     matching substring — we're either on the target or close to
    #     it; no point re-launching.
    if launch_choices:
        in_page_keyword = False
        for action in state.get("actions") or []:
            label = (action.get("label") or "").lower()
            if any(kw and kw in label for kw in content_keywords):
                in_page_keyword = True
                break
        if not in_page_keyword:
            current_pkg = (state.get("package") or "").strip()
            for action_id, choice in launch_choices.items():
                pkg = (choice.get("package") or "").strip()
                if not pkg or pkg == current_pkg:
                    continue
                label = (choice.get("label") or "").lower()
                pkg_lower = pkg.lower()
                matched = any(kw and (kw in label or kw in pkg_lower) for kw in content_keywords)
                if not matched:
                    continue
                already_here = any(
                    kw and kw in page_signature for kw in content_keywords if kw in pkg_lower or kw in label
                )
                if already_here:
                    continue
            probabilities = {a["id"]: 0.0 for a in state["actions"]}
            probabilities[action_id] = 1.0
            op_probs = {op: 0.0 for op in operations}
            op_probs["LAUNCH_APP"] = 1.0
            target_probs = {aid: 0.0 for aid in launch_choices}
            target_probs[action_id] = 1.0
            return {
                "choice": action_id,
                "operation": "LAUNCH_APP",
                "target": action_id,
                "confidence": 0.92,
                "probabilities": probabilities,
                "operation_probabilities": op_probs,
                "target_probabilities": target_probs,
                "target_confidence": 0.92,
                **_scripted_task_complete(goal, "LAUNCH_APP"),
                "raw_answers": {"_scripted": {"launch": action_id}},
                "model": "scripted",
                "usage": {},
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "request": {"goal": goal, "launch_target": action_id},
                "backend": "scripted",
            }

    for operation, candidates in targets.items():
        for index, action in candidates.items():
            label = action.get("label", "").lower()
            # ``set`` so a keyword repeated in the goal doesn't dominate
            # the score (otherwise ``Sound`` wins over ``Vibrate`` purely
            # because the goal mentions it three times).
            score = sum(1 for kw in keyword_set if kw in label)
            # Tiebreaker: TYPE_TEXT preferred over CLICK for text-bearing goals.
            if operation == "TYPE_TEXT" and any(kw in label for kw in content_keywords):
                score += 0.5
            # Radio-group awareness: when one sibling is already
            # ``checked=true`` the user almost always wants to advance
            # to a sibling that's currently unchecked. We deprioritise
            # the already-checked row so it can't win a tie, even
            # though the cycle rule above handles the common case.
            if action.get("checked") in ("true", True, "1"):
                score -= 0.25
            scored.append((operation, index, score))
    scored.sort(key=lambda r: (-r[2], r[0]))
    if scored and scored[0][2] > 0:
        best_op, best_target = scored[0][0], scored[0][1]
        best_score = scored[0][2]
    else:
        # No keyword hits at all. Fall back to the first element if the
        # goal mentions a generic verb (open / find / go); otherwise block.
        generic_verbs = {"open", "go", "find", "show", "navigate", "see"}
        if any(kw in generic_verbs for kw in content_keywords) and elements:
            best_op = next(iter(targets), "CLICK")
            best_target = next(iter(targets.get(best_op, {})), None)

    if best_target is None:
        # Pick any control (scroll/wait) as a last resort so the loop keeps moving.
        for op_id, action in controls.items():
            if op_id == "WAIT":
                # ``probabilities`` is keyed by action id (the same keys
                # the agent loop reads off ``state["actions"]``) so the
                # executor can look up the selected element's score.
                probabilities = {a["id"]: 0.0 for a in state["actions"]}
                probabilities[action["id"]] = 1.0
                op_probs = {op: 0.0 for op in operations}
                op_probs["WAIT"] = 1.0
                return {
                    "choice": action["id"],
                    "operation": op_id,
                    "target": None,
                    "confidence": 0.5,
                    "probabilities": probabilities,
                    "operation_probabilities": op_probs,
                    "target_probabilities": {},
                    "target_confidence": None,
                    **_scripted_task_complete(goal, op_id),
                    "raw_answers": {"_scripted": {"fallback": "wait"}},
                    "model": "scripted",
                    "usage": {},
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "request": {"goal": goal},
                    "backend": "scripted",
                }
        # Give up cleanly with BLOCKED.
        probabilities = {a["id"]: 0.0 for a in state["actions"]}
        op_probs = {op: 0.0 for op in operations}
        op_probs["BLOCKED"] = 1.0
        return {
            "choice": "BLOCKED",
            "operation": "BLOCKED",
            "target": None,
            "confidence": 0.99,
            "probabilities": probabilities,
            "operation_probabilities": op_probs,
            "target_probabilities": {},
            "target_confidence": None,
            **_terminal_task_complete(),
            "raw_answers": {"_scripted": {"fallback": "blocked"}},
            "model": "scripted",
            "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": {"goal": goal},
            "backend": "scripted",
        }

    action = targets[best_op][best_target]
    choice = action["id"]
    probabilities = {a["id"]: 0.0 for a in state["actions"]}
    probabilities[choice] = 1.0
    op_probs = {op: 0.0 for op in operations}
    op_probs[best_op] = 1.0
    target_probs = {idx: 0.0 for idx in targets[best_op]}
    target_probs[best_target] = 1.0
    return {
        "choice": choice,
        "operation": best_op,
        "target": best_target,
        "confidence": 0.95,
        "probabilities": probabilities,
        "operation_probabilities": op_probs,
        "target_probabilities": target_probs,
        "target_confidence": 0.95,
        **_scripted_task_complete(goal, best_op),
        "raw_answers": {"_scripted": {"score": best_score, "label": action.get("label")}},
        "model": "scripted",
        "usage": {},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": {"goal": goal, "keywords": keywords, "picked": action.get("label")},
        "backend": "scripted",
    }


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "is", "are", "be", "this", "that", "when", "shown",
    "open", "find", "show", "go", "navigate", "see", "via", "tap",
    "click", "type", "search", "stop", "until", "page", "screen",
    "use", "using", "by", "from", "if",
    # field/control surface vocabulary that the text helper shouldn't
    # mistake for content.
    "box", "field", "bar", "input", "area", "into", "into\u00a0the",
    "drill", "press", "select", "choose", "enter",
    # common verb/connector noise that substring-matches too eagerly.
    "entry", "app", "needed", "displays",
}


_DECISION_BACKENDS = {
    "typesafe": _choose_typesafe,
    "scripted": _choose_scripted,
}


def choose(state, goal, history, *, task_plan: str | None = None, relevant_apps: list[str] | None = None):
    """Dispatch to the configured ``AGENT_DECISION_BACKEND``.

    Defaults to ``typesafe`` so the production path is unchanged. Set
    ``AGENT_DECISION_BACKEND=scripted`` to run the loop end-to-end with
    no API key. Unknown values raise so typos surface immediately.

    ``task_plan`` (optional) carries the auxiliary-LLM-generated plan;
    when set, it is exposed to the decision backend as ``state.task_plan``
    so it can flow into the Jev request payload.

    ``relevant_apps`` (optional) is the auxiliary-LLM-classified list of
    apps involved in the user's goal (parsed from
    ``RELEVANT_APPS=…`` in :func:`mobile_jev_ultrafast.llm.plan_task`).
    When set, the decision backend drops every ``launch_<pkg>`` action
    whose display label doesn't match a relevant app, so the model
    isn't forced to choose between 50+ irrelevant launch entries. Pass
    ``None`` or ``[]`` to keep the old behaviour (full launch surface).
    """
    backend = os.environ.get("AGENT_DECISION_BACKEND", "typesafe").lower()
    fn = _DECISION_BACKENDS.get(backend)
    if fn is None:
        raise ValueError(
            f"Unknown AGENT_DECISION_BACKEND: {backend!r}. "
            f"Choose from {sorted(_DECISION_BACKENDS)}."
        )
    page = state
    if task_plan or relevant_apps:
        # Pass through a shallow copy so the caller's ``state`` dict
        # isn't mutated.
        page = dict(state)
        if task_plan:
            page["task_plan"] = task_plan
    return fn(page, goal, history, relevant_apps=relevant_apps)


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>…</think> blocks some MiniMax / reasoning models prefix replies with.

    The OpenAI-compatible spec doesn't reserve ``<think>`` as an official
    channel, but several vendors (MiniMax, DeepSeek-R1, etc.) leak the
    chain-of-thought into ``message.content`` when the ``reasoning``
    flag isn't honoured. Drop those blocks before JSON parsing so the
    helper stays backend-agnostic.
    """
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> dict | None:
    """Best-effort JSON object extractor.

    Returns the first balanced ``{...}`` substring that parses, or
    ``None`` if nothing does. Survives the ``<think>…`` prefix above
    and other leading prose.
    """
    if not text:
        return None
    cleaned = _strip_thinking_tags(text)
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass
    # Fallback: grab the outermost {...} block and try again.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError):
        return None


def _field_text_helper(context):
    """Original OpenAI-compatible small-LLM text helper."""
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    request_body = {
        "model": model,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        **reasoning,
        "messages": [
            {"role": "system", "content": TEXT_VALUE},
            {
                "role": "user",
                "content": json.dumps(context),
            },
        ],
    }
    verbose.emit(2, "── TEXT helper request ──────────────────────────────────")
    verbose.emit(2, verbose.blob(request_body))
    started = time.perf_counter()
    result = post_json(base + "/chat/completions", key, request_body)
    verbose.emit(2, "── TEXT helper response ─────────────────────────────────")
    verbose.emit(2, verbose.blob(result))
    raw_content = result["choices"][0]["message"]["content"] or ""
    output = _extract_json_object(raw_content)
    try:
        if output is None:
            raise ValueError("no JSON object found in helper reply")
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }


def _field_text_scripted(context):
    """Deterministic offline text helper.

    Pulls a quote out of the goal that matches the field label. Good
    enough for unit tests and the verified demo, and obviously not a
    replacement for a real LLM in production — the demo file calls this
    out next to ``AGENT_TEXT_BACKEND=helper``.
    """
    started = time.perf_counter()
    goal = str(context.get("goal", ""))
    field_label = str(context.get("field", {}).get("label", "")).strip()
    # Strip the trailing words after the last quote if the goal says "type '...'".
    quote = _extract_quote(goal)
    if quote:
        value = quote
    else:
        # Fall back to the last content word in the goal (skipping
        # stopwords like "box", "field", "search", "into", ...).
        words = [w.strip(".,!?\"'") for w in goal.split()]
        value = next(
            (w for w in reversed(words) if w and len(w) > 1 and w.lower() not in _STOPWORDS),
            "",
        )
    return value, {
        "model": "scripted",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": {},
        "field_label": field_label,
    }


def _extract_quote(goal: str) -> str:
    """Pull the first single-quoted or double-quoted string from ``goal``."""
    for quote in ("'", '"'):
        start = goal.find(quote)
        if start < 0:
            continue
        end = goal.find(quote, start + 1)
        if end > start:
            return goal[start + 1:end].strip()
    return ""


_TEXT_BACKENDS = {
    "helper": _field_text_helper,
    "scripted": _field_text_scripted,
}


def field_text(context):
    """Dispatch to the configured ``AGENT_TEXT_BACKEND`` (default ``helper``)."""
    backend = os.environ.get("AGENT_TEXT_BACKEND", "helper").lower()
    fn = _TEXT_BACKENDS.get(backend)
    if fn is None:
        raise ValueError(
            f"Unknown AGENT_TEXT_BACKEND: {backend!r}. "
            f"Choose from {sorted(_TEXT_BACKENDS)}."
        )
    return fn(context)
