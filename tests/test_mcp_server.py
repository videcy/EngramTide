"""Official MCP SDK registration contract."""

import pytest

from engramtide.hooks.claude_code import (
    POST_COMPACT_PATH,
    STOP_PATH,
    USER_PROMPT_SUBMIT_PATH,
)
from engramtide.mcp.server import mcp


@pytest.mark.asyncio
async def test_mcp_exposes_write_and_admin_tools_only():
    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "dehydrate",
        "remember",
        "forget",
        "consolidate",
        "restore_archived",
        "run_maintenance",
        "search_memories",
        "list_memories",
        "export_memories",
        "memory_stats",
        "search_archive",
    }
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.streamable_http_path == "/mcp"
    assert mcp.settings.stateless_http is True
    assert mcp.settings.transport_security.enable_dns_rebinding_protection is True


def test_hook_routes_are_served_by_the_same_app():
    app = mcp.streamable_http_app()
    routes = {
        route.path: route.methods
        for route in app.routes
        if getattr(route, "path", "").startswith("/hooks/")
    }

    assert set(routes) == {USER_PROMPT_SUBMIT_PATH, STOP_PATH, POST_COMPACT_PATH}
    assert all("POST" in methods for methods in routes.values())
