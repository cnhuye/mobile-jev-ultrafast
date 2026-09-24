"""Drive WeChat (or any a11y-blocked app) with OCR + MCP tap.

The AutoX.js MCP server reports an almost-empty UI tree for WeChat
(\"a11y-blocked\" apps collapse to a single root with ``a: \"d\"``),
so the agent's element table is useless. This example flips on the OCR
fallback the device layer ships: when ``get_ui_tree`` is too sparse,
:func:`AutoX.observe` runs ``mcp.ocr(source=\"screenshot\")`` and
synthesises a click-action table from the recognised text.

The script below is the minimal loop the *real* verified-demo path
would compose in production. It does not call the Jev model \u2014 it
finds the labelled \"\u901a\u8baf\u5f55\" tab by exact OCR match and taps its
centre coordinate, mirroring what the agent policy would do given the
same observed state.

Usage:

    uv run --env-file .env python examples/ocr_tap_wechat.py \\
        --label \"\u901a\u8baf\u5f55\"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from my_autox_server.autox import AutoX


def _save_screenshot(b64: str, out: Path) -> bool:
    import base64

    if not b64:
        return False
    out.write_bytes(base64.b64decode(b64))
    print(f"  saved {out} ({out.stat().st_size:,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("AUTOX_MCP_URL"),
        help="AutoX.js MCP endpoint (default: $AUTOX_MCP_URL).",
    )
    parser.add_argument(
        "--label",
        default="通讯录",
        help="Exact text of the control you want to tap (default: 通讯录).",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("artifacts"),
        type=Path,
        help="Where to dump screenshots + the OCR trace (default: artifacts).",
    )
    parser.add_argument(
        "--no-tap",
        action="store_true",
        help="Skip the tap; useful for diagnostics that only need the OCR dump.",
    )
    args = parser.parse_args()
    if not args.url:
        print("ERROR: --url or $AUTOX_MCP_URL is required", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== connect {args.url} with OCR fallback ===")
    device = AutoX(args.url, ocr_fallback=True)
    try:
        print("=== observe ===")
        page = device.observe(screenshot=True)
    finally:
        device.close()

    print(f"  package={page['package']!r} activity={page['activity']!r}")
    print(f"  fallback={page.get('fallback')!r} ocr_count={page.get('ocr_count', 0)}")
    print(f"  ocr_confidence_avg={page.get('ocr_confidence_avg', 0.0):.2f}")
    if page.get("fallback") != "ocr":
        print(
            "  ! UI tree already had enough elements; OCR fallback was NOT used.",
            file=sys.stderr,
        )
    real = [a for a in page["actions"] if a["id"].startswith(("e", "ocr"))]
    labels = [a["label"] for a in real]
    print(f"  visible labels ({len(labels)}): {labels[:8]}{'...' if len(labels) > 8 else ''}")

    screenshot_path = args.out_dir / "ocr_tap_screen.jpg"
    _save_screenshot(page.get("screenshot"), screenshot_path)

    target = next((a for a in real if a["label"] == args.label), None)
    if target is None:
        print(f"  ! label {args.label!r} not in OCR output; nothing to tap")
        (args.out_dir / "ocr_actions.json").write_text(
            json.dumps(real, indent=2, ensure_ascii=False)
        )
        return 1

    cx = target["rect"]["x"] + target["rect"]["w"] // 2
    cy = target["rect"]["y"] + target["rect"]["h"] // 2
    print(
        f"=== found {args.label!r} at ({cx},{cy}) "
        f"(confidence={target.get('confidence', 0.0):.2f}) ==="
    )

    if args.no_tap:
        print("  --no-tap set; skipping the tap.")
        return 0

    # Re-open a connection to actually tap; the previous one is closed.
    device = AutoX(args.url, ocr_fallback=True)
    try:
        device.mcp.tap(cx, cy)
        time.sleep(1.0)
    finally:
        device.close()
    print(f"=== tapped ({cx},{cy}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())