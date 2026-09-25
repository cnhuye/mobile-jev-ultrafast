"""The complete agent loop. Typed choices, observable state, bounded execution.

Line-for-line port of :mod:`jev_ultrafast.agent` with a single change: the
device layer is :class:`AutoX` instead of :class:`Browser`. All freshness
checks, text-helper caching, and the ``tick == predict + act`` handoff
behave identically on a phone.

This module also wires the auxiliary-LLM hooks ported from
``mobile-jev-jarvis``:

* :func:`Agent.__init__` optionally calls :func:`mobile_jev_ultrafast.llm.plan_task`
  on the first observation so Jev gets a short ``state.task_plan`` and a
  list of relevant apps before its first decision.
* :meth:`Agent._detect_deadlock` watches the rolling action window and
  either flips ``status`` to ``blocked`` or triggers the LLM deadlock
  breaker; the breaker's structured suggestion is either executed
  immediately (PRESS_BACK etc.) or injected into ``recent_actions`` as
  ``LLM_SUGGESTION`` so Jev obeys it on the next tick.
* :func:`Agent._maybe_summarize` runs after the loop ends and emits a
  one-line summary the dashboard can render.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from . import verbose
from .autox import AutoX as Browser
from .autox import StalePage
from .model import TASK_COMPLETE_FINISH_THRESHOLD, action_space, choose, field_context, field_text
from .questions import MAX_STEPS

log = logging.getLogger(__name__)

# Sliding window length for the deadlock detector. mobile-jev-jarvis
# uses 8; we mirror that so the LLM fallback prompt looks identical.
_DEADLOCK_WINDOW = 8
# Same threshold mobile-jev-jarvis uses: three identical non-scroll
# actions inside the window is enough to call the LLM.
_DEADLOCK_REPEATS = 3
# Detection for "scrolling without clicking": if the last N actions are
# all scrolls, the agent isn't making progress; surface as deadlock.
_SCROLL_ONLY_WINDOW = 5

# Standard map from LLM deadlock suggestions to direct action ids.
# Maps :func:`llm.decide_next_action` ``kind`` values onto the action
# ids the device layer recognises. ``wait_longer`` is included so the
# deadlock-decide LLM can escape splash-screen / ad stalls by waiting
# 3 s instead of repeating the click on the ad creative.
_DIRECT_ACTION_FOR_SUGGESTION = {
    "press_back": "press_back",
    "press_home": "press_home",
    "press_recents": "press_recents",
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "scroll_up": "scroll_up",
    "scroll_down": "scroll_down",
    "wait_longer": "wait_longer",
}


class Agent:
    def __init__(
        self,
        url,
        goals,
        *,
        record_dir=None,
        screenshots=False,
        device=None,
        llm_plan=False,
        llm_fallback=False,
        llm_summary=False,
        verbose_level=0,
    ):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        # ``device`` lets callers inject a custom backend (typically FakeAutoX
        # for tests / local demo). When omitted, AutoX connects to the MCP
        # server in ``AUTOX_MCP_URL``. ``url`` is kept for parity with
        # jev-ultrafast; it doubles as a human-readable device label.
        self.browser = device if device is not None else Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        self.llm_plan = llm_plan
        self.llm_fallback = llm_fallback
        self.llm_summary = llm_summary
        # ``verbose_level`` mirrors the CLI's ``-v`` count. The actual log
        # routing lives in :mod:`mobile_jev_ultrafast.verbose`; this is
        # kept so direct ``Agent`` users can inspect / bump it.
        self.verbose_level = int(verbose_level or 0)
        self.log_buffer: list[str] = []
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            task_plan="",
            plan_relevant_apps=[],
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            log_buffer=self.log_buffer,
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

        self._trace_observation(page, step=0, phase="initial")

        # Pre-task planning runs after construction so the device layer
        # has a chance to enumerate installed apps (which the planner
        # forwards to the LLM).
        if self.llm_plan:
            try:
                from . import llm

                plan_result = llm.plan_task(self.state["goal"], self.browser, enabled=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("plan_task failed during init: %s", exc)
                plan_result = {"plan_text": "", "relevant_apps": [], "model": "scripted"}
            self.state["task_plan"] = plan_result.get("plan_text", "")
            self.state["plan_relevant_apps"] = plan_result.get("relevant_apps", [])
            if self.state["task_plan"]:
                self._log("PLAN", self.state["task_plan"])

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log(self, kind: str, message: str) -> None:
        """Append a tagged line to the rolling log buffer.

        ``kind`` is the prefix Jev reads in the deadlock prompt and the
        dashboard reads in the summary prompt (``[PLAN]`` / ``[ACT]`` /
        ``[DEC]`` / ``[ERR]``).
        """
        self.state["log_buffer"].append(f"[{kind}] {message}")

    # ------------------------------------------------------------------
    # Verbose tracing
    # ------------------------------------------------------------------

    def _trace_observation(self, page: dict, *, step: int, phase: str) -> None:
        """Level-1 observation summary + level-3 element table.

        This is the single most useful line when debugging a loop: it
        shows which app/screen the agent is standing on and how many
        actionable elements it can see *before* it decides anything.
        """
        if not verbose.enabled(1):
            return
        actions = page.get("actions") or []
        actionable = [a for a in actions if a.get("kind") in {"click", "fill", "select"}]
        text = (page.get("text") or "").replace("\n", " ⏎ ")[:120]
        verbose.emit(
            1,
            f"step {step} OBSERVE({phase}): {page.get('package', '?')}/{page.get('activity', '?')} "
            f"elements={len(actions)} actionable={len(actionable)} "
            f"text={text!r}",
        )
        verbose.emit(
            3,
            f"step {step} OBSERVE({phase}) url={page.get('url')!r} "
            f"fingerprint={page.get('fingerprint', '')[:12]}",
        )
        for element in action_space(actions)[0]:
            verbose.emit(
                3,
                f"step {step}   [{element['index']:>3}] {element['role']:<12} "
                f"{element['label'][:44]!r} ops={','.join(element.get('operations', []))}",
            )

    def _trace_decision(self, decision: dict, *, step: int) -> None:
        """Level-1 one-liner + level-3 structured decision dump."""
        if not verbose.enabled(1):
            return
        verbose.emit(
            1,
            f"step {step} DECIDE: operation={decision.get('operation')} "
            f"target={decision.get('target')} choice={decision.get('choice')} "
            f"confidence={decision.get('confidence')} "
            f"latency={decision.get('latency_ms')}ms backend={decision.get('backend', '?')}",
        )
        probs = decision.get("probabilities") or {}
        if probs:
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            verbose.emit(1, "  probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        # ``task_complete`` is independent of operation; print it on its
        # own line so the trace shows whether the model thinks this step
        # is terminal, plus the configured threshold for context.
        tc = decision.get("task_complete")
        if tc is not None:
            verbose.emit(
                1,
                f"  task_complete: choice={tc} "
                f"confidence={decision.get('task_complete_confidence', 0.0):.3f} "
                f"finish_prob={decision.get('task_complete_probabilities', {}).get('finish', 0.0):.3f} "
                f"threshold={TASK_COMPLETE_FINISH_THRESHOLD}",
            )
        verbose.emit(
            3,
            f"step {step} DECIDE full: "
            + verbose.blob({k: v for k, v in decision.items() if k != 'raw_answers'}),
        )
        if decision.get("request") is not None:
            verbose.emit(3, verbose.wrap(f"step {step} DECIDE request: ", verbose.blob(decision["request"])))

    def _trace_act(self, action: dict, *, step: int, text: str | None = None) -> None:
        """Level-1 execution line emitted right before the device acts."""
        if not verbose.enabled(1):
            return
        rect = action.get("rect") or {}
        verbose.emit(
            1,
            f"step {step} ACT: {action.get('kind')} {action.get('id')} "
            f"{action.get('label')!r} @({rect.get('x')},{rect.get('y')},{rect.get('w')}x{rect.get('h')})",
        )
        if text:
            verbose.emit(1, f"step {step} ACT: text={text!r}")

    def _trace_result(self, entry: dict, *, step: int) -> None:
        """Level-1 post-action line: did the screen actually change?"""
        if not verbose.enabled(1):
            return
        verbose.emit(
            1,
            f"step {step} RESULT: page_changed={entry.get('page_changed')} "
            f"elapsed={entry.get('elapsed_ms')}ms url={entry.get('url')!r}",
        )

    def _trace_deadlock(self, deadlock: dict, *, step: int) -> None:
        if not verbose.enabled(1):
            return
        verbose.emit(
            1,
            f"step {step} DEADLOCK: kind={deadlock.get('kind')} "
            f"action_key={deadlock.get('action_key')} note={deadlock.get('note')}",
        )
        window = list(self.state["history"])[-_DEADLOCK_WINDOW:]
        verbose.emit(
            3,
            f"step {step} DEADLOCK window=",
        )
        for entry in window:
            verbose.emit(
                3,
                f"step {step}   {entry.get('step'):>3} {entry.get('kind', '?'):<10} "
                f"{entry.get('operation', '?'):<14} choice={entry.get('choice', '-')} "
                f"action={entry.get('action', '')!r} page_changed={entry.get('page_changed')}",
            )

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    def _detect_deadlock(self):
        """Decide whether the agent is stuck.

        Returns ``None`` if progress is normal, or a ``dict`` with
        ``"note"`` / ``"action_key"`` describing the deadlock. The
        caller decides whether to flip ``status`` to ``blocked`` or
        call the LLM for an escape hint.
        """
        recent = list(self.state["history"])
        if len(recent) < _DEADLOCK_REPEATS:
            return None
        # Sliding window of the last 8 non-scroll actions. Each key is
        # ``"<kind>:<id>"`` so a click on e1 and a fill on e2 are
        # distinguishable; scroll is excluded from the loop check (it
        # legitimately repeats) but counted below for the "scrolls only"
        # heuristic.
        window = recent[-_DEADLOCK_WINDOW:]
        non_scroll = [h for h in window if h.get("kind") != "scroll"]
        action_keys = [
            f"{h.get('kind', '?')}:{h.get('choice', '?')}"
            for h in non_scroll
        ]
        if len(action_keys) >= _DEADLOCK_REPEATS:
            tail = action_keys[-_DEADLOCK_REPEATS:]
            if len(set(tail)) == 1:
                return {
                    "kind": "repeat",
                    "action_key": tail[-1],
                    "note": (
                        f"Action {tail[-1]!r} has occurred "
                        f"{_DEADLOCK_REPEATS} times in the last {len(action_keys)} non-scroll steps."
                    ),
                }
        last_kinds = [h.get("kind") for h in recent[-_SCROLL_ONLY_WINDOW:]]
        if len(last_kinds) >= _SCROLL_ONLY_WINDOW and all(k == "scroll" for k in last_kinds):
            return {
                "kind": "scroll_only",
                "action_key": "scroll",
                "note": (
                    f"No click / type / key in the last {_SCROLL_ONLY_WINDOW} steps — "
                    "the agent is scrolling without finding a useful element."
                ),
            }
        return None

    def _trigger_llm_fallback(self, deadlock: dict) -> dict:
        """Ask the auxiliary LLM how to escape the loop.

        Returns the structured suggestion (parsed by ``llm.decide_next_action``).
        Falls back to the scripted suggestion when the LLM is unavailable
        so the loop never crashes.
        """
        try:
            from . import llm
        except Exception:  # noqa: BLE001
            llm = None
        if llm is None:
            return {"kind": "press_back", "text": "执行 PRESS_BACK 返回"}
        suggestion = llm.decide_next_action(
            goal=self.state["goal"],
            page=self.state["page"],
            recent_actions=list(self.state["history"][-8:]),
            deadlock_note=deadlock["note"],
            enabled=self.llm_fallback,
        )
        return suggestion

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _maybe_summarize(self) -> dict | None:
        """Generate the post-run summary if enabled."""
        if not self.llm_summary:
            return None
        try:
            from . import llm
        except Exception:  # noqa: BLE001
            return None
        return llm.summarize_result(
            goal=self.state["goal"],
            page=self.state["page"],
            logs=list(self.state["log_buffer"]),
            success=self.state["status"] == "done",
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            step = len(state["history"]) + 1
            self._trace_observation(state["page"], step=step, phase="pre-decide")
            state["decision"] = choose(
                state["page"],
                state["goal"],
                state["history"],
                task_plan=state.get("task_plan") or None,
                relevant_apps=state.get("plan_relevant_apps") or None,
            )
            self._trace_decision(state["decision"], step=step)
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            self._trace_act(action, step=len(state["history"]) + 1, text=text)
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            history_entry = {
                "step": len(state["history"]) + 1,
                "action": action["label"],
                "kind": action["kind"],
                "choice": selected,
                "probability": decision["probabilities"][selected],
                "confidence": decision["confidence"],
                "latency_ms": decision["latency_ms"],
                "text": text,
                "text_helper": helper["model"] if helper else None,
                "text_latency_ms": helper["latency_ms"] if helper else 0,
                "operation": decision["operation"],
                "target": decision["target"],
                # task_complete head — independent of ``operation``. The
                # loop trusts ``finish`` (above the configured threshold)
                # to stop right after this action executes, which is what
                # operational goals like "上滑 1 下" need.
                "task_complete": decision.get("task_complete", "continue"),
                "task_complete_confidence": decision.get("task_complete_confidence", 0.0),
                "task_complete_probabilities": decision.get("task_complete_probabilities", {}),
                "page_changed": None,
                "url": page["url"],
                "usage": decision["usage"],
                "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                "elapsed_ms": state["elapsed_ms"],
            }
            state["history"].append(history_entry)
            self._log("ACT", f"{history_entry['operation']} -> {history_entry['action']}")
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            self._trace_result(state["history"][-1], step=len(state["history"]))
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            # ``task_complete`` head — when the model marked this action as
            # terminal AND the signal clears the configured confidence
            # threshold, stop the loop right here. This is the path that
            # handles operational goals (「上滑 1 下」, "click that button")
            # where the action itself completes the task; without this the
            # model has no observable evidence to choose DONE on.
            task_complete = decision.get("task_complete", "continue")
            tc_confidence = float(decision.get("task_complete_confidence", 0.0) or 0.0)
            if task_complete == "finish" and tc_confidence >= TASK_COMPLETE_FINISH_THRESHOLD:
                state["status"] = "done"
                self._log(
                    "DEC",
                    f"task_complete=finish confidence={tc_confidence:.3f} "
                    f">= threshold {TASK_COMPLETE_FINISH_THRESHOLD}; loop ends after this action",
                )
                verbose.emit(
                    1,
                    f"step {len(state['history'])} TASK_COMPLETE: finish "
                    f"confidence={tc_confidence:.3f}; status -> done",
                )
                return self.snapshot()
            # Sliding-window deadlock detection replaces the old "three
            # identical + no page change" heuristic. When the LLM
            # fallback is off, we collapse to the old behaviour
            # immediately. When it's on, we record the warning so Jev
            # can see it on the next tick and let the LLM suggest the
            # escape.
            deadlock = self._detect_deadlock()
            if deadlock is None:
                state["status"] = "ready"
            elif self.llm_fallback:
                self._trace_deadlock(deadlock, step=len(state["history"]))
                suggestion = self._trigger_llm_fallback(deadlock)
                self._log("DEC", f"deadlock={deadlock['action_key']} suggestion={suggestion.get('kind')}")
                direct_id = _DIRECT_ACTION_FOR_SUGGESTION.get(suggestion.get("kind") or "")
                if direct_id:
                    # Execute the suggested action directly. Reuse the
                    # current page's fingerprint so act() doesn't refuse.
                    try:
                        direct_action = next(
                            (a for a in state["page"]["actions"] if a["id"] == direct_id),
                            None,
                        )
                        if direct_action is not None:
                            state["history"].append(
                                {
                                    "step": len(state["history"]) + 1,
                                    "action": direct_action["label"],
                                    "kind": direct_action["kind"],
                                    "choice": direct_id,
                                    "operation": "LLM_ACTION",
                                    "target": direct_id,
                                    "text": None,
                                    "probability": 1.0,
                                    "confidence": 1.0,
                                    "latency_ms": 0,
                                    "page_changed": None,
                                    "url": state["page"]["url"],
                                    "usage": {},
                                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                                    "elapsed_ms": state["elapsed_ms"],
                                    "note": suggestion.get("text", ""),
                                }
                            )
                            self._log("ACT", f"LLM_ACTION -> {direct_action['label']}")
                            verbose.emit(
                                1,
                                f"step {len(state['history'])} LLM_ACTION: "
                                f"{direct_action['kind']} {direct_action['id']} {direct_action['label']!r}",
                            )
                            state["browser"].act(direct_action, state["page"])
                            state["page"] = state["browser"].observe(screenshot=self.screenshots)
                            state["status"] = "ready"
                        else:
                            state["status"] = "blocked"
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Direct LLM escape failed: %s", exc)
                        state["status"] = "blocked"
                else:
                    # Inject the suggestion as recent_actions feedback so
                    # the next Jev tick obeys it.
                    state["history"].append(
                        {
                            "step": len(state["history"]) + 1,
                            "operation": "LLM_SUGGESTION",
                            "note": suggestion.get("text", "") or "(empty)",
                            "url": state["page"]["url"],
                            "elapsed_ms": state["elapsed_ms"],
                        }
                    )
                    self._log("DEC", f"LLM_SUGGESTION={suggestion.get('text','')}")
                    verbose.emit(
                        1,
                        f"step {len(state['history'])} LLM_SUGGESTION (injected for next tick): "
                        f"{suggestion.get('text', '')!r}",
                    )
                    state["status"] = "ready"
            else:
                self._trace_deadlock(deadlock, step=len(state["history"]))
                state["history"].append(
                    {
                        "step": len(state["history"]) + 1,
                        "operation": "WARN",
                        "note": deadlock["note"],
                        "url": state["page"]["url"],
                        "elapsed_ms": state["elapsed_ms"],
                    }
                )
                self._log("ERR", deadlock["note"])
                state["status"] = "blocked"
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")
        # Post-run summary runs once after the loop exits.
        summary = self._maybe_summarize()
        if summary:
            self.state["summary"] = summary
            self._log("SUMMARY", summary.get("summary", ""))

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()