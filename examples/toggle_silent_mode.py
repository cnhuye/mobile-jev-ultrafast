"""让 Jev 自己进设置、读当前铃声模式、切换一格。

用户需求：进入设置，如果是静音就切到振动，如果是振动就切到非静音（响铃）。

这条任务是**条件型决策**：Jev 必须先从 UI 树上读出「声音模式」三个
radio 里哪个是 `checked=true`，再决定点哪个。这依赖 MCP server 输出
`checked` 字段（见 `docs/REQUIREMENTS_MY_AUTOX.md`）；手机 build 已
补上该字段，本脚本就是端到端验证。

验证方式与 `jev_verified_demo.py` / jev-ultrafast 的 `flights.py` 一致：
**不采信模型的 DONE 自述**，而是用独立探针前后各读一次 AudioManager
的 ringer mode，断言"恰好前进一格"。

用法：

    # 真机（需要 AUTOX_MCP_URL + TYPESAFE_API_KEY）
    uv run --env-file .env python examples/toggle_silent_mode.py --live

    # 离线（FakeAutoX + scripted 后端，无需 key）
    AGENT_DECISION_BACKEND=scripted uv run python examples/toggle_silent_mode.py --fake
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mobile_jev_ultrafast import Agent, FakeAutoX

# 手机上三个模式的文案 -> 规范化名。UI 是中文，但 Jev 也可能看到英文，
# 所以两边都收。
MODE_ALIASES = {
    "silent": "silent",
    "静音": "silent",
    "vibrate": "vibrate",
    "振动": "vibrate",
    "ring": "ring",
    "sound": "ring",
    "响铃": "ring",
    "normal": "ring",
}

# 规格化后的循环：静音 → 振动 → 响铃 → 静音
CYCLE = ("silent", "vibrate", "ring")
CYCLE_LABELS_ZH = {"silent": "静音", "vibrate": "振动", "ring": "响铃"}

# Goal 模板。**关键**：目标模式由脚本（探针读到的当前模式 + 循环）
# 算好后写死进去，而不是让模型自己"读完当前状态再决定"。
#
# 如果写成条件式（"如果选中的是 A 就点 B"），模型每一 tick 都会重新
# 评估：把 A→B 切完之后，下一 tick 看到的是 B，规则又匹配上，会再切
# 一次。模型没有"我已经切过了"的记忆。写死目标就没有这个歧义：
# 到达 B 之后目标已满足，DONE。
#
# 这跟 jev-ultrafast 的 flights.py 一致：具体航线 / 日期由脚本算好写进
# goal，模型只负责把它执行完。
GOAL_TEMPLATE_ZH = (
    "打开设置，进入「声音和振动」。当前铃声模式是「{current}」，"
    "请把它切换成「{target}」。切换成功后立即停止。"
)
GOAL_TEMPLATE_FAKE = (
    "Open Settings, then tap Sound. "
    "The current ring mode is {current}, switch it to {target}. "
    "Stop once it is switched."
)


def next_mode(current: str | None) -> str | None:
    """循环里的下一个模式。"""
    if current not in CYCLE:
        return None
    return CYCLE[(CYCLE.index(current) + 1) % len(CYCLE)]


def detect_mode_from_page(page: dict) -> str | None:
    """从观察到的页面读出当前选中的 radio。"""
    for action in page.get("actions", []):
        if action.get("role") != "radio":
            continue
        if str(action.get("checked", "")).lower() == "true":
            label = (action.get("label") or "").strip().lower()
            return MODE_ALIASES.get(label)
    return None


def read_mode(device, *, live: bool) -> str | None:
    """独立读当前铃声音模式。

    --live 走 AudioManager 探针（不依赖 UI 树、不依赖模型）；
    --fake 直接读 FakeAutoX 的内部状态。
    """
    if live:
        return device.mcp.probe_ringer_mode()
    return getattr(device, "_ring_mode", None)


def verify(initial: str | None, final: str | None) -> dict:
    """独立断言：模式恰好前进一格。"""
    expected = next_mode(initial)
    return {
        "passed": bool(initial) and bool(final) and final == expected,
        "initial_mode": initial,
        "final_mode": final,
        "expected_mode": expected,
        "advanced_one_slot": bool(expected) and final == expected,
    }


def run(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fake:
        os.environ.setdefault("AGENT_DECISION_BACKEND", "scripted")
        os.environ.setdefault("AGENT_TEXT_BACKEND", "scripted")
        device = FakeAutoX("phone:fake", initial_ring=args.initial_ring)
    else:
        device = None  # Agent 会用 AUTOX_MCP_URL 构造 AutoX
        for required in ("AUTOX_MCP_URL", "TYPESAFE_API_KEY"):
            if not os.environ.get(required):
                raise SystemExit(f"--live 需要环境变量 {required}")

    # ---- 前置：回到桌面，读初始模式 ----
    if not args.fake:
        from mobile_jev_ultrafast import AutoX

        probe_device = AutoX(args.label)
        probe_device.mcp.run_script('"auto"; home();', name="go-home", timeout_millis=4000)
        time.sleep(1.5)
        initial = read_mode(probe_device, live=True)
        probe_device.close()
    else:
        initial = read_mode(device, live=False)

    print(f"=== 初始铃声音模式: {initial!r} ===", flush=True)
    if initial is None:
        raise SystemExit("读不到初始铃声音模式，说明探针或设备状态有问题。")
    target = next_mode(initial)
    print(
        f"=== 期望 Jev 把「{CYCLE_LABELS_ZH.get(initial, initial)}」"
        f"切到「{CYCLE_LABELS_ZH.get(target, target)}」 ===",
        flush=True,
    )

    # ---- 跑 Jev ----
    template = GOAL_TEMPLATE_FAKE if args.fake else GOAL_TEMPLATE_ZH
    goal = template.format(
        current=CYCLE_LABELS_ZH.get(initial, initial),
        target=CYCLE_LABELS_ZH.get(target, target),
    )
    print(f"=== goal: {goal} ===", flush=True)
    with Agent(
        url=args.label,
        goals=goal,
        screenshots=args.record,
        record_dir=out_dir / "frames" if args.record else None,
        device=device,
    ) as agent:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(
                f"{state['elapsed_ms']:>6} ms  "
                f"{len(state['history'])} actions  "
                f"{state['status']:>8}  "
                f"{last.get('operation', '-'):>9} -> {last.get('action', '')}",
                flush=True,
            )
            # --fake 的 scripted 后端没有 DONE 条件，我们点一次就收工。
            # 只有当页面真的展示了 radio（即已到 Audio 屏）且选中项变了
            # 才算数——否则在 Home/Settings 屏上会误判为"已切换"。
            detected = detect_mode_from_page(state["page"])
            if args.fake and detected is not None and detected != initial:
                agent.state["status"] = "done"
                break

    # ---- 后置：再读一次，独立断言 ----
    if args.fake:
        final = read_mode(device, live=False)
    else:
        from mobile_jev_ultrafast import AutoX

        probe_device = AutoX(args.label)
        final = read_mode(probe_device, live=True)
        probe_device.close()

    snapshot = agent.snapshot()
    verification = verify(initial, final)
    verification["url"] = snapshot["page"]["url"]
    verification["title"] = snapshot["page"]["title"]
    verification["goal"] = goal
    verification["decisions"] = [
        {
            "operation": d["operation"],
            "target": d.get("target"),
            "confidence": d.get("confidence"),
            "backend": d.get("backend"),
        }
        for d in snapshot["decisions"]
    ]
    (out_dir / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False))
    (out_dir / "state.json").write_text(json.dumps(snapshot, indent=2, default=str, ensure_ascii=False))

    print()
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    if not verification["passed"]:
        raise SystemExit(
            "铃声音模式没有恰好前进一格；详见 verification.json。"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", dest="fake", action="store_false", default=True,
        help="驱动真机（需要 AUTOX_MCP_URL + TYPESAFE_API_KEY）。这是默认值。",
    )
    parser.add_argument("--fake", dest="fake", action="store_true", help="离线 FakeAutoX。")
    parser.add_argument(
        "--initial-ring", choices=CYCLE, default="silent",
        help="仅 --fake 用：模拟屏的起始模式（默认 silent）。",
    )
    parser.add_argument("--label", default="phone:sound", help="trace 里显示的设备标签。")
    parser.add_argument(
        "--output", default="artifacts/toggle_silent_mode", type=Path,
        help="verification.json / state.json / frames 的输出目录。",
    )
    parser.add_argument("--record", action="store_true", help="逐步截图到 <output>/frames。")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
