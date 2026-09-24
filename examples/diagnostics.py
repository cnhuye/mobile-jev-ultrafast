"""Dump everything we know about a connected AutoX.js MCP server.

This is the first thing to run when something doesn't work: it reports
the tool list, the device profile, the foreground app, and probes a few
common failure modes (screen capture permission, missing tools, etc.).

Usage:
    uv run python examples/diagnostics.py \\
        [--url http://192.168.x.y:27190/mcp] \\
        [--out diagnostics.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from my_autox_server.mcp_client import MCPClient

CRITICAL_TOOLS = {
    "perception": {"get_ui_tree", "screenshot", "device_info"},
    "input": {"tap", "swipe"},
    "text": {"set_clip", "set_text", "run_script"},
    "app": {"app_control", "get_foreground_app", "get_current_activity"},
    "discovery": {"find_element", "find_elements"},
}


def probe(client: MCPClient) -> dict:
    """Return a structured report."""

    def safe(fn, *args):
        try:
            return {"ok": True, "value": fn(*args)}
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}

    report = {
        "url": client.url,
        "tools": client.tools,
    }

    by_group = {}
    for group, names in CRITICAL_TOOLS.items():
        by_group[group] = {name: (name in client.tools) for name in names}
    report["critical_tools"] = by_group

    report["device_info"] = safe(client.device_info)
    report["foreground_app"] = safe(client.get_foreground_app)
    report["current_activity"] = safe(client.get_current_activity)
    report["display_size"] = {"ok": True, "value": list(client.display_size())}

    # Screenshot is special — it's the one tool that's likely to fail
    # for permission reasons rather than missing-tool reasons. Surface
    # the actual error verbatim so the user can grant the right
    # permission inside AutoX.js.
    try:
        client.screenshot()
        report["screenshot"] = {"ok": True, "hint": "permission granted"}
    except RuntimeError as exc:
        message = str(exc)
        hint = None
        if "No screen capture permission" in message:
            hint = (
                "AutoX.js hasn't been granted screen capture permission. "
                "Open AutoX.js → Settings → Accessibility / Screenshot "
                "service and enable it; the first call will pop a "
                "MediaProjection dialog that needs the user's 'Start now'."
            )
        elif "Tool" in message and "not registered" in message:
            hint = "The MCP server doesn't advertise this tool."
        report["screenshot"] = {"ok": False, "error": message, "hint": hint}

    return report


def print_report(report: dict) -> None:
    print(f"AUTOX MCP diagnostics for {report['url']}\n")

    print(f"Tools advertised ({len(report['tools'])}):")
    for name in report["tools"]:
        print(f"  - {name}")
    print()

    print("Critical tools by group:")
    for group, names in report["critical_tools"].items():
        ok = sum(1 for v in names.values() if v)
        print(f"  [{group}] {ok}/{len(names)} present")
        for name, present in names.items():
            mark = "✓" if present else "✗"
            print(f"      {mark} {name}")
    print()

    info = report["device_info"]
    if info["ok"]:
        v = info["value"]
        print(f"Device: {v.get('manufacturer','?')} {v.get('model','?')} (SDK {v.get('sdkInt','?')}, locale={v.get('locale','?')})")
    fg = report["foreground_app"]
    if fg["ok"]:
        print(f"Foreground app: {fg['value']!r}")
    ca = report["current_activity"]
    if ca["ok"]:
        print(f"Current activity: {ca['value']!r}")
    ds = report["display_size"]
    if ds["ok"]:
        w, h = ds["value"]
        print(f"Display: {w}x{h}")
    print()

    ss = report["screenshot"]
    print("Screenshot probe:")
    if ss["ok"]:
        print(f"  ✓ {ss.get('hint', 'OK')}")
    else:
        print(f"  ✗ {ss['error']}")
        if ss.get("hint"):
            print(f"    ↳ {ss['hint']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("AUTOX_MCP_URL"))
    parser.add_argument("--token", default=os.environ.get("AUTOX_MCP_TOKEN") or None)
    parser.add_argument("--out", type=Path, default=None, help="Write the raw report here as JSON.")
    args = parser.parse_args()

    if not args.url:
        print("ERROR: --url or $AUTOX_MCP_URL must be set.", file=sys.stderr)
        return 2

    client = MCPClient(args.url, token=args.token)
    report = probe(client)
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Wrote raw report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
