"""Reference demo mirroring ``jev-ultrafast/examples/flights.py``.

Walks the agent from a launcher-style home screen to a target screen
and asserts the final state independently of the model's ``DONE``
answer. The original ``flights.py`` does the same for Google Flights:
it decodes the URL, checks the date, and verifies the visible flight
cards. This file is the phone equivalent.

Two modes:

* **Default** ``--fake`` (offline, no API key required): the agent drives
  :class:`FakeAutoX` with the deterministic ``AGENT_DECISION_BACKEND=scripted``
  backend. The whole task runs without a network round-trip and is the
  recommended way to verify the loop before plugging in a real device.
* ``--live`` (needs ``AUTOX_MCP_URL``, ``TYPESAFE_API_KEY``,
  ``TEXT_MODEL_API_KEY``): the agent drives a real phone through MCP
  and asks Jev for every decision. Output is identical to the
  ``--fake`` mode so diffing the two tells you whether the Jev model
  introduced extra steps vs. the deterministic one.

Usage:

    uv run python examples/jev_verified_demo.py --fake
    uv run --env-file .env python examples/jev_verified_demo.py --live \\
        --output artifacts/verified_demo_live
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mobile_jev_ultrafast import Agent, FakeAutoX

# Goal chosen so the scripted backend can satisfy it deterministically:
# the *last* content noun ("about") appears in the URL of the target
# screen, which is exactly the assertion ``verify()`` checks below.
# ``Stop on About`` is what the scripted backend keys off; ``Stop when
# shown`` would put the stopword "shown" at the end and the loop would
# never reach DONE.
GOAL = (
    "Open Settings, then tap About. Stop on About."
)


def verify(page: dict) -> dict:
    """Independent checks on the resulting page, not the model's DONE answer.

    Mirrors ``flights.py``'s ``verify()`` function: every assertion reads
    the observed ``page`` and compares it against known invariants of
    the target screen. If ``passed`` is False, the demo raises
    :class:`SystemExit` so CI can pick it up.
    """
    url = page["url"].lower()
    title = page["title"].lower()
    text = page["text"].lower()
    checks = {
        "url_has_about": "about" in url,
        "title_has_about": "about" in title,
        "text_has_version": any(
            indicator in text
            for indicator in ("version", "build", "0.1.0", "mobile-jev-ultrafast")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "url": page["url"],
        "title": page["title"],
    }


def run(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fake:
        os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
        os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
        device = FakeAutoX("phone:fake")
    else:
        # The Agent defaults to ``AutoX(url)`` which connects to
        # ``$AUTOX_MCP_URL``; we don't inject a device here so the
        # production path is exercised verbatim.
        device = None
        for required in ("AUTOX_MCP_URL", "TYPESAFE_API_KEY", "TEXT_MODEL_API_KEY"):
            if not os.environ.get(required):
                raise SystemExit(
                    f"--live requires {required} in the environment."
                )

    with Agent(
        url=args.label,
        goals=GOAL,
        screenshots=args.record,
        record_dir=out_dir / "frames" if args.record else None,
        device=device,
    ) as agent:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(
                f"{state['elapsed_ms']:>5} ms  "
                f"{len(state['history'])} actions  "
                f"{state['status']:>8}  "
                f"{last.get('operation', '-')} -> {last.get('action', '')}",
                flush=True,
            )

    snapshot = agent.snapshot()
    verification = verify(snapshot["page"])
    (out_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    (out_dir / "state.json").write_text(json.dumps(snapshot, indent=2, default=str))
    print()
    print(json.dumps(verification, indent=2))
    if not verification["passed"]:
        raise SystemExit(
            "Final screen did not satisfy the About-screen invariants; "
            "see verification.json for details."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake",
        action="store_true",
        default=True,
        help="Use FakeAutoX + scripted backend (default, no API key).",
    )
    parser.add_argument(
        "--live",
        dest="fake",
        action="store_false",
        help="Drive a real phone (needs AUTOX_MCP_URL + API keys).",
    )
    parser.add_argument(
        "--label",
        default="phone:settings_about",
        help="Free-form device label; only used for trace identification.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/verified_demo",
        type=Path,
        help="Where to write verification.json / state.json / frames.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Capture per-step screenshots to <output>/frames.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())