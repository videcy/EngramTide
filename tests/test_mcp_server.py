"""Official MCP SDK registration contract."""

import pytest

from engramtide.mcp.server import mcp


@pytest.mark.asyncio
async def test_mcp_exposes_expected_tools_and_safe_transport_defaults():
    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "start_session",
        "prepare_turn",
        "commit_turn",
        "end_session",
        "close_session",
        "list_memories",
        "export_memories",
        "delete_memories",
        "memory_stats",
        "search_archive",
        "run_maintenance",
    }
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.streamable_http_path == "/mcp"
    assert mcp.settings.stateless_http is True
    assert mcp.settings.transport_security.enable_dns_rebinding_protection is True
