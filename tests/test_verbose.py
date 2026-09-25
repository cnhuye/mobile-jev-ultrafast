"""Tests for the ``-v`` verbose tracing added on top of the agent loop.

Everything runs offline: the fake device + scripted decision backend for
the CLI-level assertions, and a monkeypatched ``post_json`` for the
payload-dump assertions.
"""

from __future__ import annotations

import json

import pytest

from mobile_jev_ultrafast import cli, verbose


@pytest.fixture(autouse=True)
def scripted(monkeypatch):
    monkeypatch.setenv("AGENT_DECISION_BACKEND", "scripted")
    monkeypatch.setenv("AGENT_TEXT_BACKEND", "scripted")
    yield
    # Never leave the shared logger hot between tests.
    verbose.configure(0)


# ---------------------------------------------------------------------------
# verbose module primitives
# ---------------------------------------------------------------------------

def test_configure_levels():
    verbose.configure(0)
    assert not verbose.enabled(1)
    assert not verbose.enabled(2)

    verbose.configure(1)
    assert verbose.enabled(1)
    assert not verbose.enabled(2)

    verbose.configure(2)
    assert verbose.enabled(1)
    assert verbose.enabled(2)

    verbose.configure(3)
    assert verbose.enabled(3)


def test_configure_does_not_duplicate_handlers():
    verbose.configure(1)
    first = len(verbose.logger.handlers)
    verbose.configure(2)
    second = len(verbose.logger.handlers)
    assert first == second == 1


def test_redact_masks_secrets_recursively():
    payload = {
        "model": "jev",
        "api_key": "super-secret",
        "nested": {"authorization": "Bearer abc", "token": "t", "keep": "visible"},
        "items": [{"password": "p"}, {"keep": 1}],
    }
    clean = verbose.redact(payload)
    assert clean["api_key"] == "***"
    assert clean["model"] == "jev"
    assert clean["nested"]["authorization"] == "***"
    assert clean["nested"]["token"] == "***"
    assert clean["nested"]["keep"] == "visible"
    assert clean["items"][0]["password"] == "***"
    assert clean["items"][1]["keep"] == 1
    # The original is untouched (deep copy).
    assert payload["api_key"] == "super-secret"


def test_blob_truncates_long_payloads():
    text = verbose.blob({"k": "x" * 10000})
    assert "truncated" in text


def test_blob_full_limit_keeps_retained_payload_visible():
    """The Jev dumps raise the cap past the default so the retained
    fields actually appear in the trace; the marker should only kick
    in when the rendered payload really exceeds ``BLOB_FULL_LIMIT``.
    """
    payload = {"goal": "Open Settings", "rules": "x" * 6000}
    text = verbose.blob(payload, limit=verbose.BLOB_FULL_LIMIT)
    assert "truncated" not in text
    assert '"rules"' in text


def test_blob_omit_paths_removes_nested_keys():
    payload = {
        "model": "jev-latest",
        "state": {
            "page": {"url": "com.example.app/Home"},
            "elements": [{"index": "1", "label": "Foo"}, {"index": "2", "label": "Bar"}],
            "recent_actions": [{"action": "click Foo"}],
        },
        "questions": {"operation": "choose"},
    }
    text = verbose.blob(payload, omit_paths=("state.elements", "state.recent_actions"))
    parsed = json.loads(text)
    assert "elements" not in parsed["state"]
    assert "recent_actions" not in parsed["state"]
    assert parsed["state"]["page"]["url"] == "com.example.app/Home"
    assert parsed["questions"]["operation"] == "choose"
    # The caller's data stays intact (the helper deep-copies first).
    assert payload["state"]["elements"] == [
        {"index": "1", "label": "Foo"},
        {"index": "2", "label": "Bar"},
    ]


def test_blob_omit_paths_supports_list_indices():
    payload = {"items": [{"keep": "a"}, {"drop": "b"}, {"keep": "c"}]}
    text = verbose.blob(payload, omit_paths=("items.1",))
    parsed = json.loads(text)
    assert parsed["items"] == [{"keep": "a"}, {"keep": "c"}]


def test_blob_omit_paths_ignores_missing_paths():
    payload = {"a": 1, "b": 2}
    text = verbose.blob(payload, omit_paths=("c.d", "a.b.c"))
    parsed = json.loads(text)
    assert parsed == {"a": 1, "b": 2}


def test_blob_omit_paths_leaves_empty_parent_when_leaf_removed():
    """Removing the only key from a dict leaves an empty container behind.

    The current ``_drop_path`` walker can never escalate that to a
    ``None`` result, so the dump renders ``{}`` and the absence of
    useful data is obvious from the empty JSON.
    """
    text = verbose.blob({"only": "data"}, omit_paths=("only",))
    assert text == "{}"


def test_blob_omit_paths_still_redacts_secrets():
    payload = {"api_key": "secret", "keep": "visible"}
    text = verbose.blob(payload, omit_paths=())
    parsed = json.loads(text)
    assert parsed["api_key"] == "***"
    assert parsed["keep"] == "visible"


def test_wrap_indents_continuation_lines():
    out = verbose.wrap("P: ", "one\ntwo")
    assert out.splitlines() == ["P: one", "   two"]


# ---------------------------------------------------------------------------
# CLI trace levels
# ---------------------------------------------------------------------------

def test_cli_without_verbose_is_quiet_on_stderr(capsys):
    code = cli.main(["--fake", "--no-env", "Open Settings and tap About"])
    captured = capsys.readouterr()
    assert code == 0
    assert "OBSERVE" not in captured.err
    assert "DECIDE" not in captured.err


def test_cli_v_traces_observation_decision_and_action(capsys):
    code = cli.main(["--fake", "--no-env", "-v", "Open Settings and tap About"])
    captured = capsys.readouterr()
    assert code == 0
    assert "step 0 OBSERVE(initial)" in captured.err
    assert "OBSERVE(pre-decide)" in captured.err
    assert "DECIDE: operation=CLICK" in captured.err
    assert "ACT: click" in captured.err
    assert "RESULT: page_changed=" in captured.err
    # Level 3 dumps must not appear at -v.
    assert "DECIDE full:" not in captured.err


def test_cli_vv_does_not_leak_element_table(capsys):
    cli.main(["--fake", "--no-env", "-vv", "Open Settings and tap About"])
    captured = capsys.readouterr()
    assert "OBSERVE(pre-decide)" in captured.err
    # The element table is a level-3 dump and must stay hidden at -vv.
    assert "── JEV element table ──" not in captured.err
    assert "OBSERVE(pre-decide) url=" not in captured.err


def test_cli_vvv_dumps_element_table_and_decisions(capsys):
    cli.main(["--fake", "--no-env", "-vvv", "Open Settings and tap About"])
    captured = capsys.readouterr()
    assert "OBSERVE(pre-decide) url=" in captured.err
    assert "DECIDE full:" in captured.err
    assert "textbox" in captured.err


def test_cli_verbose_keeps_stdout_clean_for_json(capsys):
    code = cli.main(["--fake", "--no-env", "--json", "--quiet", "-v", "Open Settings and tap About"])
    captured = capsys.readouterr()
    assert code == 0
    # stdout is still a single JSON document.
    payload = json.loads(captured.out)
    assert payload["status"] == "done"
    # The trace went to stderr instead.
    assert "OBSERVE" in captured.err


def test_cli_quiet_still_shows_explicit_trace(capsys):
    cli.main(["--fake", "--no-env", "--quiet", "-v", "Open Settings and tap About"])
    captured = capsys.readouterr()
    # Explicit -v is honoured even with --quiet…
    assert "OBSERVE" in captured.err
    # …while the per-step line for intermediate ``ready`` states is not
    # printed (only the terminal state + final summary survive).
    assert "ready" not in captured.out


def test_cli_verbose_flag_is_repeatable():
    args = cli.build_parser().parse_args(["-vvv", "goal"])
    assert args.verbose == 3
    args = cli.build_parser().parse_args(["-v", "-v", "goal"])
    assert args.verbose == 2
    args = cli.build_parser().parse_args(["goal"])
    assert args.verbose == 0


# ---------------------------------------------------------------------------
# model payload dumps
# ---------------------------------------------------------------------------

def test_typesafe_logs_full_request_and_response(capsys, monkeypatch):
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast import model
    from mobile_jev_ultrafast.autox import FakeAutoX

    monkeypatch.setenv("AGENT_DECISION_BACKEND", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-key")

    def fake_post_json(url, key, body):
        assert "TYPESAFE_API_KEY" not in body  # the key travels in the header
        questions = body["questions"]
        op_ids = list(questions["operation"]["criteria"])
        target_ids = list(questions["click_target"]["criteria"])
        chosen = target_ids[0]
        answers = {
            "operation": {
                "choice": "CLICK",
                "probabilities": {op: (1.0 if op == "CLICK" else 0.0) for op in op_ids},
                "confidence": 0.9,
            },
            "click_target": {
                "choice": chosen,
                "probabilities": {t: (1.0 if t == chosen else 0.0) for t in target_ids},
                "confidence": 0.8,
            },
        }
        return {"answers": answers, "model": "jev-latest", "usage": {}}

    monkeypatch.setattr(model, "post_json", fake_post_json)
    verbose.configure(2)

    fake = FakeAutoX()
    with agent_module.Agent("demo://x", "Open Settings and tap About", device=fake, verbose_level=2) as agent:
        agent.command("predict")
    err = capsys.readouterr().err
    assert "── JEV request ──" in err
    assert "── JEV response ──" in err
    assert "JEV: operation=CLICK" in err
    # Level-2 does not print the element table (that's level 3).
    assert "── JEV element table ──" not in err


def test_typesafe_level_two_omits_bulky_payload_slices(capsys, monkeypatch):
    """``-vv`` Jev dumps must drop ``state.elements`` / ``state.recent_actions``
    from the request and ``answers.click_target`` from the reply so the
    rest of the trace stays readable; level 3 still prints the element
    table separately so no data is actually lost.
    """
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast import model
    from mobile_jev_ultrafast.autox import FakeAutoX

    monkeypatch.setenv("AGENT_DECISION_BACKEND", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-key")

    request_blob: dict = {}
    response_blob: dict = {}

    def fake_post_json(url, key, body):
        request_blob.update(body)
        questions = body["questions"]
        op_ids = list(questions["operation"]["criteria"])
        target_ids = list(questions["click_target"]["criteria"])
        chosen = target_ids[0]
        answers = {
            "operation": {
                "choice": "CLICK",
                "probabilities": {op: (1.0 if op == "CLICK" else 0.0) for op in op_ids},
                "confidence": 0.9,
            },
            "click_target": {
                "choice": chosen,
                "probabilities": {t: (1.0 if t == chosen else 0.0) for t in target_ids},
                "confidence": 0.8,
            },
        }
        response_blob.update({"answers": answers, "model": "jev-latest", "usage": {}})
        return response_blob

    monkeypatch.setattr(model, "post_json", fake_post_json)
    verbose.configure(2)

    fake = FakeAutoX()
    with agent_module.Agent("demo://x", "Open Settings and tap About", device=fake, verbose_level=2) as agent:
        agent.command("predict")
    err = capsys.readouterr().err

    request_section = err.split("── JEV response ──", 1)[0]
    response_section = err.split("── JEV response ──", 1)[1]

    # Sanity: the model did receive the bulky slices on the wire (we
    # only dropped them from the *log*).
    assert "elements" in request_blob["state"]
    assert "recent_actions" in request_blob["state"]
    assert "click_target" in response_blob["answers"]

    # The level-2 request dump omits the bulky state slices.
    assert '"elements"' not in request_section
    assert '"recent_actions"' not in request_section
    # …but the parts that matter for debugging still ride along.
    assert '"model"' in request_section
    assert '"page"' in request_section
    assert '"questions"' in request_section

    # The level-2 response dump omits the bulky target distribution.
    assert '"click_target"' not in response_section
    # …but the rest of the answer set is still visible.
    assert '"operation"' in response_section
    assert '"answers"' in response_section

    # The dumped blobs were not truncated by the default 4 kB cap; the
    # raised ``BLOB_FULL_LIMIT`` lets the retained payload survive in
    # full so the parts that matter for debug actually appear.
    assert "truncated" not in request_section
    assert "truncated" not in response_section


# ---------------------------------------------------------------------------
# deadlock + LLM fallback traces
# ---------------------------------------------------------------------------

def _scroll_entry(step: int) -> dict:
    return {
        "step": step,
        "kind": "scroll",
        "choice": "scroll_up",
        "operation": "SCROLL_UP",
        "action": "向上滚动",
        "page_changed": False,
        "url": "com.example.app/Home",
    }


def test_deadlock_trace_reports_scroll_only(capsys):
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast.autox import FakeAutoX

    verbose.configure(1)
    fake = FakeAutoX()
    agent = agent_module.Agent("demo://x", "scroll the list", device=fake, verbose_level=1)
    try:
        for i in range(1, 6):
            agent.state["history"].append(_scroll_entry(i))
        deadlock = agent._detect_deadlock()
        assert deadlock is not None and deadlock["kind"] == "scroll_only"
        agent._trace_deadlock(deadlock, step=5)
        err = capsys.readouterr().err
        assert "DEADLOCK: kind=scroll_only" in err
    finally:
        agent.close()


def test_deadlock_trace_dumps_window_at_level_three(capsys):
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast.autox import FakeAutoX

    verbose.configure(3)
    fake = FakeAutoX()
    agent = agent_module.Agent("demo://x", "scroll the list", device=fake, verbose_level=3)
    try:
        for i in range(1, 6):
            agent.state["history"].append(_scroll_entry(i))
        deadlock = agent._detect_deadlock()
        agent._trace_deadlock(deadlock, step=5)
        err = capsys.readouterr().err
        assert "DEADLOCK window=" in err
        assert "SCROLL_UP" in err
    finally:
        agent.close()


def test_llm_fallback_traces_prompt_and_reply(capsys, monkeypatch):
    from mobile_jev_ultrafast import llm

    verbose.configure(2)

    seen: list[dict] = []

    def fake_post_json(url, key, body):
        seen.append(body)
        return {"choices": [{"message": {"content": "执行 PRESS_BACK 返回上一级"}}], "usage": {}}

    monkeypatch.setattr(llm, "post_json", fake_post_json)
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "secret")
    monkeypatch.setenv("TEXT_MODEL", "MiniMax-M3")
    result = llm.decide_next_action(
        goal="手机屏幕上滑1下",
        page={"package": "com.android.launcher3", "text": "桌面", "actions": []},
        recent_actions=[_scroll_entry(i) for i in range(1, 9)],
        deadlock_note="No click / type / key in the last 5 steps",
        enabled=True,
    )
    err = capsys.readouterr().err
    # The system prompt must name the allowed escape verbs, and the user
    # message must carry the deadlock reason, so the trace is actionable.
    system = seen[0]["messages"][0]["content"]
    user = seen[0]["messages"][1]["content"]
    assert "PRESS_BACK" in system
    assert "卡死原因" in user
    assert result["kind"] == "press_back"
    assert "auxiliary LLM request" in err
    assert "auxiliary LLM response" in err
    assert "执行 PRESS_BACK 返回上一级" in err
    assert "FALLBACK: parsed suggestion=press_back" in err


def test_llm_action_escape_is_traced(capsys, monkeypatch):
    """Reproduces the user's ``scroll → deadlock → press_home`` loop.

    Forces the policy to always pick ``scroll_up``; after five scrolls the
    scroll-only detector fires, the (mocked) fallback LLM suggests
    ``PRESS_HOME``, and the agent executes it. The trace must name every
    one of those transitions so a stuck run is diagnosable from stderr.
    """
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast.autox import FakeAutoX

    verbose.configure(1)

    def fake_choose(page, goal, history, *, task_plan=None, relevant_apps=None):
        return {
            "choice": "scroll_up",
            "operation": "SCROLL_UP",
            "target": None,
            "confidence": 0.6,
            "probabilities": {"scroll_up": 1.0},
            "operation_probabilities": {"SCROLL_UP": 1.0},
            "target_probabilities": {},
            "raw_answers": {},
            "model": "test",
            "usage": {},
            "latency_ms": 1,
            "request": {"goal": goal},
            "backend": "test",
        }

    monkeypatch.setattr(agent_module, "choose", fake_choose)
    fake = FakeAutoX()
    agent = agent_module.Agent(
        "demo://x", "手机屏幕上滑1下", device=fake, llm_fallback=True, verbose_level=1
    )
    monkeypatch.setattr(
        agent, "_trigger_llm_fallback", lambda deadlock: {"kind": "press_home", "text": "执行 PRESS_HOME 回桌面"}
    )
    try:
        # Four scrolls means the next one completes the scroll-only window.
        for i in range(1, 5):
            agent.state["history"].append(_scroll_entry(i))
        agent.command("tick")
        err = capsys.readouterr().err
        assert "DEADLOCK: kind=scroll_only" in err
        assert "LLM_ACTION: key press_home" in err
    finally:
        agent.close()


def test_plan_task_traces_relevant_apps(capsys, monkeypatch):
    from mobile_jev_ultrafast import llm

    verbose.configure(1)

    def fake_post_json(url, key, body):
        return {
            "choices": [{"message": {"content": "1. 回到桌面\n2. 打开设置\nRELEVANT_APPS=设置,电话"}}],
            "usage": {},
        }

    monkeypatch.setattr(llm, "post_json", fake_post_json)
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "secret")
    monkeypatch.setenv("TEXT_MODEL", "MiniMax-M3")

    class _Device:
        def list_installed_apps(self):
            return [{"label": "设置", "package": "com.android.settings"}]

    result = llm.plan_task("打开设置", _Device(), enabled=True)
    err = capsys.readouterr().err
    assert result["relevant_apps"] == ["设置", "电话"]
    assert "PLAN: relevant_apps" in err
    assert "回到桌面" in err


def test_typesafe_redacts_api_key_at_level_three(capsys, monkeypatch):
    from mobile_jev_ultrafast import agent as agent_module
    from mobile_jev_ultrafast import model
    from mobile_jev_ultrafast.autox import FakeAutoX

    monkeypatch.setenv("AGENT_DECISION_BACKEND", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-key")

    def fake_post_json(url, key, body):
        questions = body["questions"]
        op_ids = list(questions["operation"]["criteria"])
        target_ids = list(questions["click_target"]["criteria"])
        chosen = target_ids[0]
        return {
            "answers": {
                "operation": {
                    "choice": "CLICK",
                    "probabilities": {op: (1.0 if op == "CLICK" else 0.0) for op in op_ids},
                    "confidence": 0.9,
                },
                "click_target": {
                    "choice": chosen,
                    "probabilities": {t: (1.0 if t == chosen else 0.0) for t in target_ids},
                    "confidence": 0.8,
                },
            },
            "model": "jev-latest",
            "usage": {},
        }

    monkeypatch.setattr(model, "post_json", fake_post_json)
    verbose.configure(3)

    fake = FakeAutoX()
    with agent_module.Agent("demo://x", "Open Settings and tap About", device=fake, verbose_level=3) as agent:
        agent.command("predict")
    err = capsys.readouterr().err
    assert "── JEV element table ──" in err
    # ``model``/``state`` are in the dump but no credential ever is.
    assert "secret-key" not in err
