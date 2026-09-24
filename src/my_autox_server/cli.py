"""One-shot CLI: hand Jev a sentence, watch it drive the phone.

    uv run autox-run "打开设置，进入声音和振动，把静音切换成振动"
    uv run autox-run --fake "Open Settings and tap About"
    uv run autox-run --show-elements "打开微信，切到通讯录"

Unlike :mod:`my_autox_server.demo` (a browser inspector) this is a plain
command-line loop: it prints one line per decision, exits non-zero when
the run blocks or errors, and can dump machine-readable state for CI.

Environment is read from ``.env`` in the current directory (or any
parent) before anything else, so you don't need ``uv run --env-file``.
Pass ``--no-env`` to skip that and use the process environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import Agent
from .autox import AutoX, FakeAutoX
from .questions import MAX_STEPS

# ---------------------------------------------------------------------------
# .env loading (hand-rolled: no extra dependency, handles the shapes we use)
# ---------------------------------------------------------------------------

def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export ") :]
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return (key, value) if key else None


def find_env_file(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for a ``.env``."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(*, override: bool = False, start: Path | None = None) -> Path | None:
    """Load ``.env`` into ``os.environ``. Returns the file used, if any."""
    path = find_env_file(start)
    if path is None:
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autox-run",
        description="Send one natural-language goal to Jev; let it drive the phone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  autox-run \"打开设置，把铃声模式切换成振动\"\n"
            "  autox-run --show-elements \"Open WeChat and open Contacts\"\n"
            "  autox-run --fake \"Open Settings and tap About\"\n"
            "  autox-run --json \"...\" > run.json\n"
        ),
    )
    parser.add_argument(
        "goal_words",
        nargs="*",
        metavar="GOAL",
        help="Goal / instruction. Multiple words are joined with spaces. "
        "Omit to read the goal from stdin.",
    )
    parser.add_argument(
        "--goal", "-g",
        dest="goals",
        action="append",
        default=[],
        metavar="TEXT",
        help="Repeat for an ordered multi-step plan.",
    )
    parser.add_argument(
        "--label",
        default="phone",
        help="Free-form device label used in traces (default: phone).",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use the in-memory FakeAutoX instead of a real phone.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable the OCR fallback for a11y-blocked apps.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.35,
        metavar="SECONDS",
        help="Pause after each tap so the next observation isn't premature (default: 0.35).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help=f"Override the action budget (default: {MAX_STEPS}).",
    )
    parser.add_argument(
        "--show-elements",
        action="store_true",
        help="Print the initial element table before running.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save a screenshot per step into DIR.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump the final state as JSON to stdout instead of the pretty log.",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="Do not auto-load .env.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser


def resolve_goal(args) -> str:
    parts: list[str] = []
    if args.goals:
        parts = list(args.goals)
    elif args.goal_words:
        parts = [" ".join(args.goal_words)]
    if not parts:
        # Read from stdin so long goals can be piped in.
        piped = sys.stdin.read().strip()
        if piped:
            parts = [piped]
    if not parts:
        raise SystemExit("No goal given. Pass it as an argument or pipe it in.")
    return "\n".join(p.strip() for p in parts if p.strip())


def make_device(args):
    if args.fake:
        return FakeAutoX(args.label)
    try:
        return AutoX(args.label, ocr_fallback=args.ocr, settle_s=args.settle)
    except (RuntimeError, OSError) as exc:
        raise SystemExit(
            f"Could not reach the phone: {exc}\n"
            "Set AUTOX_MCP_URL (e.g. http://192.168.2.7:27190/mcp in .env) "
            "or pass --fake to run offline."
        ) from exc


def print_elements(agent: Agent) -> None:
    elements = agent.snapshot()["elements"]
    if not elements:
        print("  (element table is empty)")
        return
    for element in elements:
        bits = [f"[{element['index']:>3}]", f"{element['role']:<12}"]
        bits.append(f"{element['label'][:44]!r}")
        if element.get("value"):
            bits.append(f"value={element['value'][:24]!r}")
        for key in ("checked", "selected", "expanded"):
            if element.get(key) is not None:
                bits.append(f"{key}={element[key]}")
        bits.append("ops=" + ",".join(element.get("operations", [])))
        print("  " + "  ".join(bits))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env_file = None if args.no_env else load_env()
    if args.max_steps is not None:
        # ``agent.py`` does ``from .questions import MAX_STEPS``, so it
        # holds its own binding — patch the module the loop reads.
        from . import agent as agent_module

        agent_module.MAX_STEPS = args.max_steps

    goal = resolve_goal(args)

    if args.max_steps is None:
        args.max_steps = MAX_STEPS

    if not args.json and not args.quiet:
        if env_file:
            print(f"env:    {env_file}")
        if not args.fake:
            print(f"device: {os.environ.get('AUTOX_MCP_URL', '(AUTOX_MCP_URL unset)')}")
        else:
            print("device: FakeAutoX (offline)")
        print(f"goal:   {goal}")
        print()

    device = make_device(args)
    agent = Agent(
        url=args.label,
        goals=goal,
        device=device,
        screenshots=args.record is not None,
        record_dir=args.record,
    )
    started = len(agent.state["history"])
    loop_error: str | None = None
    try:
        if args.show_elements and not args.json:
            print("initial element table:")
            print_elements(agent)
            print()

        for state in agent.run():
            if args.json:
                continue
            last = state["history"][-1] if state["history"] else {}
            step = len(state["history"])
            mark = {"done": "✓", "blocked": "✗"}.get(state["status"], " ")
            if args.quiet and state["status"] not in {"done", "blocked"}:
                continue
            print(
                f"{step:>3} {mark} {state['elapsed_ms']:>6} ms  "
                f"{state['status']:>8}  "
                f"{last.get('operation', '-'):>9} -> {last.get('action', '')}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        agent.close()
        return 130
    except (ValueError, RuntimeError) as exc:
        # The loop raises when it runs out of action budget or when the
        # model/device misbehaves. Report it as a failed run rather than
        # dumping a traceback.
        loop_error = str(exc)
    finally:
        snapshot = agent.snapshot()
        agent.close()

    status = snapshot["status"]
    if loop_error:
        status = "blocked"
        if not args.json:
            print(f"\nrun stopped: {loop_error}", file=sys.stderr)
    if args.json:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
    else:
        print()
        print(f"status: {status}")
        print(f"steps:  {len(snapshot['history']) - started}")
        print(f"final:  {snapshot['page']['url']}")
        if snapshot.get("decisions"):
            ops = [d["operation"] for d in snapshot["decisions"]]
            print(f"ops:    {' -> '.join(ops)}")

    if status == "done":
        return 0
    if status == "blocked":
        print("run blocked; the goal was not completed", file=sys.stderr)
        return 1
    print(f"run ended in unexpected state: {status}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
