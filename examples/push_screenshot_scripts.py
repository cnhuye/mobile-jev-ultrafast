"""Push helper scripts to AutoX.js and trigger the screen capture prompt.

AutoX.js cannot obtain a MediaProjection token by itself — it must start
a screen capture, which causes Android to show a system dialog asking
the user to allow it. This script writes two files to AutoX's scripts
directory, then runs the one that requests capture so the prompt shows
up.

Run with:

    uv run python examples/push_screenshot_scripts.py \\
        --url http://192.168.2.7:27190/mcp

After the dialog appears, tap **"立即开始" / "Start now"** on the phone;
screenshot over MCP will start working.

The two files written to AutoX's scripts dir are:
    enable-screen-capture.js   one-shot, triggers the MediaProjection prompt
    take-screenshot.js         snap a PNG into /sdcard/Pictures
"""

from __future__ import annotations

import argparse
import os
import sys

from my_autox_server.mcp_client import MCPClient

ENABLE_SCREEN_CAPTURE = r'''"auto";
// Trigger Android's MediaProjection prompt. After the user accepts,
// AutoX.js v7 caches the token in CaptureForegroundService and any
// subsequent screenshot, ocr, find_image, ... request works.
toastLog("Requesting screen capture permission...");
try {
    images.requestScreenCapture();
    toastLog("Screen capture ready. Tap back / close this to confirm.");
} catch (e) {
    toastLog("requestScreenCapture failed: " + e);
}
'''


TAKE_SCREENSHOT = r'''"auto";
// Snap a PNG into /sdcard/Pictures/screenshot-{timestamp}.png.
// Requires that the user already accepted the MediaProjection prompt
// at least once (see enable-screen-capture.js).
const now = new Date();
const stamp = now.getFullYear() + pad2(now.getMonth() + 1) + pad2(now.getDate())
            + "-" + pad2(now.getHours()) + pad2(now.getMinutes()) + pad2(now.getSeconds());
function pad2(n) { return n < 10 ? "0" + n : "" + n; }

const path = "/sdcard/Pictures/screenshot-" + stamp + ".png";
const img = images.captureScreen();
if (!img) {
    toastLog("captureScreen returned null. Did you accept MediaProjection?");
    exit();
}
images.save(img, path);
toastLog("Saved " + path);
'''


def save_script(client: MCPClient, *, name: str, script: str) -> dict:
    return client.call_tool(
        "save_script",
        {"name": name, "script": script, "overwrite": True},
    )


def run_script(client: MCPClient, *, script: str, name: str | None = None) -> dict:
    args = {"script": script, "mode": "v7", "timeoutMillis": 15000}
    if name:
        args["name"] = name
    return client.call_tool("run_script", args)


def list_scripts(client: MCPClient) -> list:
    try:
        result = client.call_tool("list_scripts", {"limit": 50})
    except RuntimeError as exc:
        print(f"  (list_scripts failed: {exc})", file=sys.stderr)
        return []
    data = client._extract(result)
    if isinstance(data, dict) and isinstance(data.get("scripts"), list):
        return data["scripts"]
    if isinstance(data, list):
        return data
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("AUTOX_MCP_URL"))
    parser.add_argument("--token", default=os.environ.get("AUTOX_MCP_TOKEN") or None)
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Push the scripts but do not auto-run the capture prompt.",
    )
    args = parser.parse_args()
    if not args.url:
        print("ERROR: --url (or $AUTOX_MCP_URL) is required.", file=sys.stderr)
        return 2

    client = MCPClient(args.url, token=args.token)
    print(f"Connected to {args.url}")

    scripts = list_scripts(client)
    existing = {s.get("name") if isinstance(s, dict) else s for s in scripts}
    print(f"AutoX script dir has {len(scripts)} file(s).")
    for s in scripts[:10]:
        print(f"  - {s}")

    print("\n→ save_script: enable-screen-capture.js")
    save_script(client, name="enable-screen-capture.js", script=ENABLE_SCREEN_CAPTURE)

    print("→ save_script: take-screenshot.js")
    save_script(client, name="take-screenshot.js", script=TAKE_SCREENSHOT)

    scripts = list_scripts(client)
    after = {s.get("name") if isinstance(s, dict) else s for s in scripts}
    for n in ("enable-screen-capture.js", "take-screenshot.js"):
        present = n in after
        print(f"  {'✓' if present else '✗'} {n}")

    if args.no_run:
        print("\n(--no-run set; skipping run_script.)")
        return 0

    print("\n→ run_script: enable-screen-capture (triggers MediaProjection prompt)")
    run_script(client, script=ENABLE_SCREEN_CAPTURE, name="enable-screen-capture")
    print("  Watch the phone — Android should pop a system dialog.")
    print("  Tap **「立即开始」** (or your OEM's equivalent) to grant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
