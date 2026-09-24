"""Capture the current screen from an AutoX.js MCP server.

Usage:
    uv run --env-file .env python examples/screenshot.py \\
        [--url http://192.168.x.y:27190/mcp] [--out phone.jpg]

Defaults to ``$AUTOX_MCP_URL``. Saves the JPEG/PNG the MCP server returns
to ``--out`` (default ``phone.jpg`` next to ``.env``) and prints the file
size. Safe to run repeatedly; each capture is a single round trip.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from mobile_jev_ultrafast.mcp_client import MCPClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("AUTOX_MCP_URL"),
        help="AutoX.js MCP endpoint (default: $AUTOX_MCP_URL).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AUTOX_MCP_TOKEN") or None,
        help="X-Token header value (default: $AUTOX_MCP_TOKEN).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("phone.jpg"),
        help="Where to write the screenshot (default: phone.jpg).",
    )
    parser.add_argument(
        "--no-base64",
        action="store_true",
        help="Ask the server for a file path instead of inline base64.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Just list available MCP tools and exit.",
    )
    args = parser.parse_args()

    if not args.url:
        print(
            "ERROR: --url or $AUTOX_MCP_URL must be set, e.g.\n"
            "  http://192.168.2.100:27190/mcp",
            file=sys.stderr,
        )
        return 2

    print(f"Connecting to {args.url} ...")
    client = MCPClient(args.url, token=args.token)

    if args.list_tools:
        print("Tools advertised by the server:")
        for name in client.tools:
            print(f"  - {name}")
        return 0

    print("Calling screenshot(asBase64=True) ...")
    try:
        b64 = client.screenshot(as_base64=True)
    except RuntimeError as exc:
        print(f"screenshot failed: {exc}", file=sys.stderr)
        return 1

    if not b64:
        print("Server returned no image data.", file=sys.stderr)
        return 1

    raw = base64.b64decode(b64)
    args.out.write_bytes(raw)
    print(f"Wrote {args.out} ({len(raw):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
