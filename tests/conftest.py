"""Make ``import my_autox_server`` work under ``uv run pytest``.

The package lives in ``src/my_autox_server``. ``pytest.ini`` already pins
``rootdir = .`` so test collection stays inside this project; we just
need the import path to resolve.
"""

import sys
from pathlib import Path


def _bootstrap():
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap()
