"""Tests for the ``autox-run`` command-line entry point.

Everything here runs offline: ``--fake`` swaps in ``FakeAutoX`` and the
scripted decision backend needs no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_autox_server import cli

# ---------------------------------------------------------------------------
# .env handling
# ---------------------------------------------------------------------------

def test_parse_env_line_handles_the_usual_shapes():
    assert cli._parse_env_line("FOO=bar") == ("FOO", "bar")
    assert cli._parse_env_line("export FOO=bar") == ("FOO", "bar")
    assert cli._parse_env_line('FOO="bar baz"') == ("FOO", "bar baz")
    assert cli._parse_env_line("FOO='bar'") == ("FOO", "bar")
    assert cli._parse_env_line("  FOO = bar  ") == ("FOO", "bar")
    assert cli._parse_env_line("# comment") is None
    assert cli._parse_env_line("") is None
    assert cli._parse_env_line("no equals sign") is None


def test_find_env_file_walks_up_from_nested_dir(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert cli.find_env_file(nested) == tmp_path / ".env"


def test_find_env_file_returns_none_when_absent(tmp_path):
    assert cli.find_env_file(tmp_path) is None


def test_load_env_does_not_clobber_existing_values(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CLI_TEST_KEY=from_file\n")
    monkeypatch.setenv("CLI_TEST_KEY", "from_shell")
    cli.load_env(start=tmp_path)
    assert __import__("os").environ["CLI_TEST_KEY"] == "from_shell"
    cli.load_env(start=tmp_path, override=True)
    assert __import__("os").environ["CLI_TEST_KEY"] == "from_file"


# ---------------------------------------------------------------------------
# goal resolution
# ---------------------------------------------------------------------------

def test_resolve_goal_joins_positional_words(monkeypatch):
    args = cli.build_parser().parse_args(["Open", "Settings", "--fake"])
    assert cli.resolve_goal(args) == "Open Settings"


def test_resolve_goal_uses_repeatable_flag_for_plans():
    args = cli.build_parser().parse_args(["-g", "Open Settings", "-g", "Tap About"])
    assert cli.resolve_goal(args) == "Open Settings\nTap About"


def test_resolve_goal_reads_stdin_when_nothing_given(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("piped goal\n"))
    args = cli.build_parser().parse_args([])
    assert cli.resolve_goal(args) == "piped goal"


# ---------------------------------------------------------------------------
# end-to-end (offline)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def scripted(monkeypatch):
    monkeypatch.setenv("AGENT_DECISION_BACKEND", "scripted")
    monkeypatch.setenv("AGENT_TEXT_BACKEND", "scripted")


def test_cli_fake_run_reaches_the_target_screen(capsys):
    code = cli.main(["--fake", "--no-env", "Open Settings and tap About"])
    out = capsys.readouterr().out
    assert code == 0
    assert "status: done" in out
    assert "AboutActivity" in out


def test_cli_json_mode_emits_parseable_state(capsys):
    import json

    code = cli.main(["--fake", "--no-env", "--json", "--quiet", "Open Settings and tap About"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "done"
    assert payload["page"]["activity"].endswith("AboutActivity")
    # The machine-readable dump must carry the decision trace.
    assert [d["operation"] for d in payload["decisions"]][-1] == "DONE"


def test_cli_show_elements_prints_the_initial_table(capsys):
    cli.main(["--fake", "--no-env", "--show-elements", "Open Settings and tap About"])
    out = capsys.readouterr().out
    assert "initial element table:" in out
    assert "Open Settings" in out


def test_cli_blocked_run_exits_nonzero(capsys):
    # No element matches, so the scripted backend gives up.
    code = cli.main(["--fake", "--no-env", "--quiet", "zzzz qqqq"])
    assert code == 1
    assert "blocked" in capsys.readouterr().err


def test_cli_missing_goal_is_rejected(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(SystemExit):
        cli.main(["--fake", "--no-env"])


def test_cli_max_steps_is_honoured(monkeypatch, capsys):
    from my_autox_server import agent as agent_module

    original = agent_module.MAX_STEPS
    try:
        cli.main(["--fake", "--no-env", "--max-steps", "1", "--quiet", "Open Settings and tap About"])
        assert agent_module.MAX_STEPS == 1
    finally:
        agent_module.MAX_STEPS = original


def test_cli_reports_unreachable_phone(monkeypatch, capsys):
    monkeypatch.delenv("AUTOX_MCP_URL", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--no-env", "open settings"])
    assert "Could not reach the phone" in str(excinfo.value)


def test_env_file_flag_is_reported(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("AUTOX_MCP_URL=http://x/mcp\n")
    monkeypatch.chdir(tmp_path)
    cli.main(["--fake", "Open Settings"])
    assert ".env" in capsys.readouterr().out


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.fake is False
    assert args.ocr is False
    assert args.settle == pytest.approx(0.35)
    assert args.no_env is False
    assert Path(args.label).name == "phone"
