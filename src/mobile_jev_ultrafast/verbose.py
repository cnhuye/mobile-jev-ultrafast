"""Verbose tracing helpers.

Everything the agent wants to show when the user asks "what is going on"
funnels through a single logger (``mobile_jev_ultrafast.verbose``) so the
CLI can turn the firehose on with ``-v`` / ``-vv`` / ``-vvv`` without
touching the loop.

Levels (mirrors the ``-v`` count on ``autox-run``):

* ``1`` — one line per decision: the observation summary (package /
  activity / element count), what Jev chose, what was executed, whether
  the screen changed, and any deadlock / LLM-fallback activity.
* ``2`` — the payloads exchanged with the models: the Jev request body
  (with ``state.elements`` / ``state.recent_actions`` dropped so the
  rest stays readable) and its raw answer (with the bulky
  ``answers.click_target`` probability map dropped), plus the
  auxiliary-LLM system prompt and reply for plan / deadlock / summary.
* ``3`` — full state dumps: the element table, the recent-action window,
  and the structured decision dict.

The logger writes to stderr only; stdout stays reserved for the per-step
line and ``--json`` output.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable

#: The one logger the whole package uses for verbose traces.
logger = logging.getLogger("mobile_jev_ultrafast.verbose")

#: The configured verbosity (the CLI's ``-v`` count). Tracked separately
#: from the logger level because Python's logging module only has three
#: coarse levels; we need to gate level 3 out at ``-vv``.
_active_level = 0

#: Message text past this many characters is truncated in level-1 output.
_MAX_BLOB = 4000

#: Higher cap for the Jev request / response dumps. Even after
#: ``state.elements`` + ``state.recent_actions`` + ``answers.click_target``
#: are dropped the body still runs ~25 kB on a normal page (the
#: ``questions.instructions.rules`` text alone is a few kB), and the
#: user wants the retained payload to stay readable for debug at
#: ``-vv``. Generous enough to fit realistic requests without
#: truncation; pathological pages still hit the marker.
BLOB_FULL_LIMIT = 50_000

#: Sentinel returned by :func:`_drop_path` to signal the caller should
#: remove its reference (vs. returning the unchanged node when the path
#: did not match).
_OMIT = object()


def configure(verbosity: int, *, stream=None) -> None:
    """Wire the verbose logger up for ``verbosity`` (0 disables it).

    The ``--quiet`` CLI flag only silences the pretty per-step line; an
    explicit ``-v`` always gets its trace, so ``--quiet -vv`` remains
    usable for debugging.
    """
    import sys

    global _active_level
    _active_level = max(0, int(verbosity or 0))
    logger.handlers.clear()
    logger.propagate = False
    if _active_level <= 0:
        logger.setLevel(logging.CRITICAL + 1)
        return
    # Level 1+ needs INFO; level 2+ needs DEBUG. Level 3 is gated in
    # :func:`enabled` / :func:`emit`, not by the logger itself.
    logger.setLevel(logging.INFO if _active_level == 1 else logging.DEBUG)
    handler = logging.StreamHandler(stream or sys.stderr)
    # Keep the formatter dumb: callers already prefix each line with its
    # step, so tests can assert on plain substrings.
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def enabled(level: int) -> bool:
    """True when a message at ``level`` would be emitted."""
    return _active_level >= level


def emit(level: int, message: str) -> None:
    """Log ``message`` at verbose ``level`` (1 => INFO, 2+ => DEBUG)."""
    if _active_level < level:
        return
    if level <= 1:
        logger.info(message)
    else:
        logger.debug(message)


def redact(value):
    """Return a deep copy of ``value`` with obviously-secret fields masked.

    Verbose level 3 dumps whole request bodies; those carry the TypeSafe /
    OpenAI API keys. Redaction keeps the trace useful without leaking
    credentials into logs, terminals, or CI attachments.
    """
    secret_keys = {
        "api_key", "apikey", "authorization", "token", "secret",
        "password", "key",
    }

    def scrub(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if isinstance(k, str) and k.lower() in secret_keys:
                    out[k] = "***"
                else:
                    out[k] = scrub(v)
            return out
        if isinstance(node, list):
            return [scrub(item) for item in node]
        return node

    return scrub(copy.deepcopy(value))


def _drop_path(node, segments):
    """Return ``node`` with the sub-path described by ``segments`` removed.

    ``segments`` is the dot-split path tail. The recursive walk mutates a
    copy of ``node`` so the caller's data stays intact; if the path runs
    past a leaf, ``_OMIT`` is returned so the parent knows to drop its
    reference. Dict keys that aren't present and list indices that fall
    outside the current bounds leave the tree untouched.
    """
    if not segments:
        return _OMIT
    head, tail = segments[0], segments[1:]
    if isinstance(node, dict):
        if head not in node:
            return node
        new_child = _drop_path(node[head], tail)
        if new_child is _OMIT:
            return {k: v for k, v in node.items() if k != head}
        return {**node, head: new_child}
    if isinstance(node, list):
        try:
            idx = int(head)
        except ValueError:
            return node
        if not 0 <= idx < len(node):
            return node
        new_child = _drop_path(node[idx], tail)
        if new_child is _OMIT:
            return node[:idx] + node[idx + 1:]
        return node[:idx] + [new_child] + node[idx + 1:]
    return node


def _omit_paths(value, paths):
    """Return a deep copy of ``value`` with each dotted path removed.

    Paths use dot-separated segments; numeric segments index into lists
    (so ``"items.0"`` reaches the first element of ``items``). Non
    matching paths are silently ignored. Removing a leaf leaves an empty
    parent behind (e.g. dropping ``"only"`` from ``{"only": "data"}``
    returns ``{}``) — callers can spot the empty container in the dump.
    """
    out = copy.deepcopy(value)
    for path in paths:
        out = _drop_path(out, path.split("."))
        if out is _OMIT:
            return None
    return out


def blob(value, *, limit: int = _MAX_BLOB, omit_paths: Iterable[str] = ()) -> str:
    """Pretty-print ``value`` as JSON with secrets redacted + paths omitted + capped.

    ``omit_paths`` is an iterable of dotted paths (``"state.elements"``,
    ``"answers.click_target"``, ``"items.0"``). Each path is removed
    from the deep-copied value before serialising, which keeps verbose
    traces compact without hiding the parts that matter for debugging.
    Numeric segments index into lists; non-existent paths are silently
    ignored so call sites don't have to defend against partial data.
    """
    cleaned = redact(value)
    if omit_paths:
        cleaned = _omit_paths(cleaned, list(omit_paths))
        if cleaned is None:
            return "<omitted>"
    try:
        text = json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        return text[:limit] + f"\n… ({len(text) - limit} more chars truncated)"
    return text


def wrap(prefix: str, text: str) -> str:
    """Indent a multi-line blob under a ``prefix`` marker for readability."""
    lines = text.splitlines() or [""]
    out = [f"{prefix}{lines[0]}"]
    out.extend(" " * len(prefix) + line for line in lines[1:])
    return "\n".join(out)
