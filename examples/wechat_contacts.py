"""Open WeChat on the phone and switch to the Contacts (通讯录) tab.

Run with:

    uv run python examples/wechat_contacts.py \\
        --url http://192.168.2.7:27190/mcp

This script demonstrates the *minimal-viable-device-control* path we
have working today, given the constraints discovered during
real-phone bring-up:

* The MCP server is exposed at ``$AUTOX_MCP_URL`` and the MCP tap/screenshot
  tools work (the phone already has MediaProjection auth'd + the
  AutoX "Share via" dialog resolves correctly).
* MCP ``tap`` is the only reliable input — it goes through
  ``runtime.automator.click`` and needs the AutoX accessibility service
  enabled (the system-share dialog disappeared after a normal launch
  the moment we used ``tap`` on a system-dialog "Cancel" button).
* ``shell input tap`` is *not* available on this phone (no root +
  SELinux-restricted), so the script doesn't bother with it.

The script:

1. Brings WeChat to the front (or launches it).
2. Sleeps 2 s for the splash.
3. Drains HUAWEI's "使用以下方式打开" dialog by tapping the
   "取消" button if it appears.
4. Taps the **通讯录** tab (centre column of 4 tabs on the bottom nav).
5. Snapshots the screen to ``artifacts/wechat_contacts.jpg``.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

from mobile_jev_ultrafast.mcp_client import MCPClient

WECHAT_PACKAGE = "com.tencent.mm"

# Bottom-nav tab centres on a 1080-wide, 2340-tall screen. Image scale = 332/1080.
TAB_X_CHAT, TAB_X_CONTACTS, TAB_X_DISCOVER, TAB_X_ME = 135, 405, 675, 945
TAB_Y = 2242

# HUAWEI System-Share dialog's "取消" button (centre).
DIALOG_CANCEL_X = 540
DIALOG_CANCEL_Y = 2208


def save_b64(b64: str, out: Path) -> bool:
    if not b64:
        return False
    out.write_bytes(base64.b64decode(b64))
    print(f"  saved {out} ({out.stat().st_size:,} bytes)")
    return True


def foreground(client: MCPClient) -> tuple[str | None, str | None]:
    """Return (packageName, activity) or (None, None) on error."""
    try:
        result = client.call_tool("get_foreground_app", {})
    except RuntimeError:
        return None, None
    data = client._extract(result)
    if isinstance(data, dict):
        return data.get("packageName"), data.get("activity")
    return None, None


def node_text_in_tree(client: MCPClient, needle: str) -> bool:
    try:
        result = client.call_tool("get_ui_tree", {})
    except RuntimeError:
        return False
    data = client._extract(result)
    if not isinstance(data, dict):
        return False

    def walk(n):
        if needle in (n.get("t") or "") or needle in (n.get("d") or ""):
            return True
        for c in n.get("children") or []:
            if walk(c):
                return True
        return False

    return walk(data)


def dismiss_share_dialog(client: MCPClient) -> bool:
    """If the HUAWEI "use which app to open" dialog is on top, tap 取消."""
    if not node_text_in_tree(client, "使用以下方式打开"):
        return False
    print("  detected HUAWEI share dialog — tapping 取消")
    client.call_tool("tap", {"x": DIALOG_CANCEL_X, "y": DIALOG_CANCEL_Y})
    time.sleep(1.5)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("AUTOX_MCP_URL"))
    parser.add_argument("--token", default=os.environ.get("AUTOX_MCP_TOKEN") or None)
    parser.add_argument("--out-dir", default="artifacts", type=Path)
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Skip the launch step (assume WeChat is already on screen).",
    )
    args = parser.parse_args()
    if not args.url:
        print("ERROR: --url or $AUTOX_MCP_URL is required", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = MCPClient(args.url, token=args.token)

    if not args.no_launch:
        print(f"=== launch {WECHAT_PACKAGE} ===")
        client.call_tool(
            "app_control",
            {"action": "launch", "packageName": WECHAT_PACKAGE},
        )
        time.sleep(2.5)

    # Step: drain HUAWEI system dialog if present.
    print("=== check for system dialog ===")
    for _ in range(3):
        if not dismiss_share_dialog(client):
            break

    # Snapshot the home screen for the record.
    print("=== snapshot home ===")
    b64 = client.screenshot()
    home_path = args.out_dir / "wechat_home.jpg"
    if save_b64(b64, home_path):
        print(f"  ({home_path})")

    print(f"=== tap 通讯录 tab at ({TAB_X_CONTACTS},{TAB_Y}) ===")
    client.call_tool(
        "tap", {"x": TAB_X_CONTACTS, "y": TAB_Y}
    )
    time.sleep(2.0)

    # Drain dialog again in case a second one popped.
    dismiss_share_dialog(client)
    time.sleep(1.0)

    print("=== snapshot contacts ===")
    b64 = client.screenshot()
    contacts_path = args.out_dir / "wechat_contacts.jpg"
    save_b64(b64, contacts_path)

    print("=== verify ===")
    pkg, activity = foreground(client)
    print(f"  foreground: packageName={pkg!r}, activity={activity!r}")

    indicators = ["新的朋友", "仅聊天的朋友", "群聊", "标签", "公众号", "服务号"]
    matches = [k for k in indicators if node_text_in_tree(client, k)]
    print(f"  contacts-page indicators seen in tree: {matches}")
    if len(matches) >= 3:
        print("  ✓ contacts tab confirmed")
    elif contacts_path.exists():
        print(f"  ! text-tree indicators thin, but screenshot saved to {contacts_path}")
        print("    (UI Automator may not have populated yet for third-party apps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
