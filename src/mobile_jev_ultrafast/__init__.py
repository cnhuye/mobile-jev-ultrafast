"""Jev chooses an observed phone action. Code owns MCP execution.

The control flow mirrors jev-ultrafast (``agent.py`` / ``model.py`` /
``questions.py``) almost verbatim. Only the device layer differs: a
browser harness over CDP is replaced by an AutoX.js client over MCP.
``Agent`` runs the same predict-then-act loop with the same freshness
guarantees; the model never sees selectors, coordinates, or executable
shell code.
"""

from .agent import Agent
from .autox import AutoX, FakeAutoX, StalePage
from .mcp_client import MCPClient

__all__ = ["Agent", "AutoX", "FakeAutoX", "MCPClient", "StalePage"]
