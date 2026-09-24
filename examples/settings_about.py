"""End-to-end example: walk from Settings → About on a real Android phone.

Matches the recipe in ``docs/CLIENT_INTEGRATION.md`` §8.1. Run with:

    AGENT_USE_FAKE=0 uv run --env-file .env python examples/settings_about.py
"""

from __future__ import annotations

import argparse

from mobile_jev_ultrafast import Agent, FakeAutoX, MCPClient

GOAL = (
    "Open the Settings app, scroll if needed, and tap the 'About phone' "
    "entry. Stop when the screen displays the Android version."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=None,
        help="AutoX.js MCP base URL (defaults to AUTOX_MCP_URL).",
    )
    parser.add_argument(
        "--fake", action="store_true", help="Use FakeAutoX instead of a real device."
    )
    args = parser.parse_args()

    if args.fake:
        device = FakeAutoX("phone:fake")
    else:
        mcp = MCPClient.from_env()
        if args.device:
            mcp.url = args.device
        device = None  # Agent will construct AutoX with the env-driven MCP client.

    with Agent(args.device or "phone:settings", GOAL, device=device) as agent:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(
                f"{state['elapsed_ms']:>5} ms  "
                f"{len(state['history'])} actions  "
                f"{state['status']:>8}  "
                f"{last.get('action', '')}",
                flush=True,
            )

    verification = {
        "url": agent.state["page"]["url"],
        "title": agent.state["page"]["title"],
    }
    print()
    print("Stopped at:", verification)
    if "About" not in verification["title"] and "about" not in verification["url"].lower():
        raise SystemExit("Final screen was not the About screen; the model may be stuck.")


if __name__ == "__main__":
    main()
