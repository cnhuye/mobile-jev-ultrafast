"""Deprecated shim — use the ``autox-run`` console script instead.

    uv run autox-run "打开设置，进入声音和振动，把模式切换成振动"
    uv run autox-run --help

The console script adds ``.env`` auto-loading, ``--fake`` / ``--ocr`` /
``--json`` / ``--max-steps`` and a non-zero exit code when the run blocks.
This file is kept so older invocations keep working:

    uv run python examples/run.py --label demo://x --goal "Open Settings"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_autox_server.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
