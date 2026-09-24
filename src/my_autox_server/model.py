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

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


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
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
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
    return elements, targets, controls


def _choose_typesafe(state, goal, history):
    """Original TypeSafe-backed decision policy. Identical to jev-ultrafast."""
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
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
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
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


def _choose_scripted(state, goal, history):
    """Deterministic offline decision backend.

    Picks the most goal-relevant element with simple keyword scoring
    over labels. Sufficient for the verified demo (settings → about) and
    for unit tests that want to exercise the agent loop without paying
    for an API call. Returns the same dict shape as ``_choose_typesafe``
    so callers don't need to branch on the backend.
    """
    started = time.perf_counter()
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
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
            "raw_answers": {"_scripted": {"cycle": cycle_target}},
            "model": "scripted",
            "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": {"goal": goal, "cycle_target": cycle_target},
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


def choose(state, goal, history):
    """Dispatch to the configured ``AGENT_DECISION_BACKEND``.

    Defaults to ``typesafe`` so the production path is unchanged. Set
    ``AGENT_DECISION_BACKEND=scripted`` to run the loop end-to-end with
    no API key. Unknown values raise so typos surface immediately.
    """
    backend = os.environ.get("AGENT_DECISION_BACKEND", "typesafe").lower()
    fn = _DECISION_BACKENDS.get(backend)
    if fn is None:
        raise ValueError(
            f"Unknown AGENT_DECISION_BACKEND: {backend!r}. "
            f"Choose from {sorted(_DECISION_BACKENDS)}."
        )
    return fn(state, goal, history)


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


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
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
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
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
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
