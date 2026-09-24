"""End-to-end example: search for a product in a shopping app.

Companion to ``docs/CLIENT_INTEGRATION.md`` §8.2. Run with:

    AGENT_USE_FAKE=0 uv run --env-file .env python examples/search_query.py \\
        --query "iPhone 15 case"
"""

from __future__ import annotations

import argparse

from mobile_jev_ultrafast import Agent, FakeAutoX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="iPhone 15 case", help="Search term to type.")
    parser.add_argument("--app", default="com.taobao.taobao", help="App package id to open.")
    parser.add_argument("--fake", action="store_true", help="Use FakeAutoX instead of a real device.")
    args = parser.parse_args()

    device = FakeAutoX("phone:fake") if args.fake else None
    goal = (
        f"Open {args.app}, tap the search box, type '{args.query}' and tap the search "
        f"button. Stop when matching result cards are visible on the screen."
    )

    with Agent(label=args.app, goal=goal, device=device) as agent:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(
                f"{state['elapsed_ms']:>5} ms  "
                f"{len(state['history'])} actions  "
                f"{state['status']:>8}  "
                f"{last.get('action', '')}  "
                f"{last.get('text') or ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
