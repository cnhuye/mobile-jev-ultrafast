# mobile-jev-ultrafast

**English** · [简体中文](README.zh-CN.md)

**Give it a sentence. It drives the phone.**

A phone agent built on [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast)'s
architecture, with the device layer swapped from the Chrome DevTools Protocol
to [AutoX.js](https://github.com/cnhuye/AutoX)'s MCP server.

[TypeSafe's Jev](https://docs.typesafe.ai/introduction) picks **one operation and one
element** from an indexed table of what is on screen. Code turns that choice into a tap.
The model never emits a coordinate, a selector, or a snippet of executable script.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Phone side](https://img.shields.io/badge/phone%20side-cnhuye%2FAutoX-orange)](https://github.com/cnhuye/AutoX)

---

## What it is

Most "LLM controls a phone" setups ask the model to produce a tap coordinate or an
XPath-like selector. Both are guesses that go stale between the screenshot and the
tap.

This project asks a different question instead:

> Given the elements currently on screen, **which one** should be acted on, and
> **how**?

The phone's accessibility tree is flattened into a numbered element table, Jev answers
`operation + target` in a single request, and the executor resolves that target back to
live screen bounds immediately before touching anything. If the screen moved in the
meantime, the action is rejected and the loop re-observes.

```
  phone screen                     indexed element table                one Jev request
  ────────────                     ──────────────────────               ───────────────
  get_ui_tree  ──────────►  [1] button   Settings
                            [2] button   Sound & vibration
                            [3] textbox  Search settings
                            [4] radio    Ring         checked
                            [5] radio    Vibrate
                            [6] radio    Silent                   ┌──────────────────┐
                            [scroll_down] [scroll_up] [wait]  ─────►│ operation        │
                                                                    │ click_target     │
                                                                    │ type_text_target │
                                                                    └────────┬─────────┘
                                                                             │
                                                        CLICK [5] ───────────┴───► tap(center of [5])
                                                        TYPE_TEXT [3] "wifi" ───► tap + setText
```

Two properties fall out of this design, and they are the whole point:

- **Every action is grounded in an observation.** The model can only pick from
  targets that were actually on screen when it decided.
- **The executor re-validates before every input.** A decision that no longer
  matches the screen raises `StalePage` instead of tapping the wrong thing.

## Quick start

```bash
git clone git@github.com:cnhuye/mobile-jev-ultrafast.git
cd mobile-jev-ultrafast
uv sync
cp .env.example .env      # fill in AUTOX_MCP_URL and TYPESAFE_API_KEY
```

Then hand it a sentence:

```bash
uv run autox-run "Open Settings, then open Sound & vibration and pick Vibrate."
```

```
env:    /path/to/mobile-jev-ultrafast/.env
device: http://192.168.2.7:27190/mcp
goal:   Open Settings, then open Sound & vibration and pick Vibrate.

  1     2526 ms     ready      CLICK -> Settings
  2     3791 ms     ready      CLICK -> Sound & vibration
  3     5955 ms     ready      CLICK -> Vibrate
  3 ✓   6622 ms      done      CLICK -> Vibrate

status: done
steps:  3
ops:    CLICK -> CLICK -> CLICK -> DONE
```

`autox-run` prints one line per decision and exits non-zero when the run blocks, so it
drops straight into a shell script or CI job.

## Command line

```bash
uv run autox-run "Open Settings and turn on Airplane mode"   # one goal
uv run autox-run -g "Open Settings" -g "Tap About phone"      # an ordered plan
echo "Open Settings and tap About phone" | uv run autox-run   # goal from stdin
```

| Flag | What it does |
|------|--------------|
| `--show-elements` | print the indexed element table Jev is choosing from |
| `-v` / `-vv` / `-vvv` | per-decision trace / + full Jev + aux-LLM payloads / + element-table & deadlock-window dumps (stderr) |
| `--ocr` | enable the OCR fallback for apps that block accessibility (WeChat, …) |
| `--max-steps N` | raise or lower the action budget (default 60) |
| `--settle S` | pause after each tap so the next observation isn't premature (default 0.35s) |
| `--record DIR` | save a screenshot after every step |
| `--json` | dump the final state as JSON — pipe it into `jq` |
| `--quiet` | summary only |

`.env` is loaded automatically from the current directory or any parent. Use
`--no-env` to opt out.

Verbose tracing goes to **stderr**, so `--json` keeps stdout parseable:

```bash
uv run autox-run -v  "scroll up once"      # observation → decision → action per step
uv run autox-run -vv "scroll up once" 2>trace.log   # + full Jev request/response
```

### Inspect a run

```bash
uv run autox        # browser UI on http://127.0.0.1:8767
```

Step through decisions one at a time, watch the element table change, and toggle the
target overlay to see exactly which rectangles the model was offered.

## The phone side

The executor talks to **[cnhuye/AutoX](https://github.com/cnhuye/AutoX)** — a fork of
AutoX.js v7 with an MCP server bolted on. It is a normal Android app; install it on the
phone you want to drive.

That app is what actually does the work on the device:

- exposes `get_ui_tree`, `tap`, `swipe`, `screenshot`, `ocr`, `run_script`, … over
  JSON-RPC on port `27190`
- dispatches taps through Android's accessibility service
- captures the screen through MediaProjection
- runs OCR on-device with Google ML Kit

### Setting it up

1. Install `cnhuye/AutoX` on the phone, then open **Settings → MCP service** and enable
   it. Point it at `0.0.0.0` so your machine can reach it, and note the port
   (default `27190`).
2. Enable the **accessibility service**. Without it the MCP server accepts `tap` and
   returns `ok`, but nothing actually moves on screen.
3. Grant **screen capture**. The first `screenshot` call pops a MediaProjection dialog;
   tap "Start now". This is a one-shot grant and will need redoing after the app is
   reinstalled.
4. Point `.env` at it:

   ```bash
   AUTOX_MCP_URL=http://192.168.2.7:27190/mcp
   AUTOX_MCP_TOKEN=          # only if you set one in the app
   ```

Check the link before blaming the model:

```bash
uv run python examples/diagnostics.py
```

It reports the tool list, device profile, foreground app, and probes screenshot
permissions, calling out the fix for whatever is missing.

## Requirements

- Python ≥ 3.12 and [`uv`](https://github.com/astral-sh/uv)
- An Android phone running [cnhuye/AutoX](https://github.com/cnhuye/AutoX) with the MCP
  service enabled
- A [TypeSafe](https://docs.typesafe.ai/introduction) API key for the decision model
- Optionally, an OpenAI-compatible endpoint for the small text helper, used only when
  the chosen operation is `TYPE_TEXT`

## Credits

- [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) — the
  control loop, the dynamic operation/target policy, and the prompts. `agent.py`,
  `model.py`, and `questions.py` are direct ports; only the device layer differs.
- [`cnhuye/AutoX`](https://github.com/cnhuye/AutoX) — the Android side.
- [TypeSafe](https://docs.typesafe.ai/introduction) — the Jev model.

## License

MIT
