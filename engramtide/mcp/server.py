"""Official MCP SDK server exposing EngramTide over Streamable HTTP."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from engramtide import EngramTide
from engramtide.mcp.manager import EngramTideMCPManager


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


HOST = os.getenv("ENGRAMTIDE_MCP_HOST", "127.0.0.1").strip()
PORT = _env_int("ENGRAMTIDE_MCP_PORT", 8765)
PATH = os.getenv("ENGRAMTIDE_MCP_PATH", "/mcp").strip()
if not PATH.startswith("/"):
    raise ValueError("ENGRAMTIDE_MCP_PATH must start with '/'")


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


_allowed_hosts = _csv_env("ENGRAMTIDE_MCP_ALLOWED_HOSTS")
_allowed_origins = _csv_env("ENGRAMTIDE_MCP_ALLOWED_ORIGINS")
if HOST not in {"127.0.0.1", "localhost", "::1"} and not _allowed_hosts:
    raise ValueError(
        "ENGRAMTIDE_MCP_ALLOWED_HOSTS is required when listening beyond localhost"
    )
transport_security = (
    TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    )
    if _allowed_hosts or _allowed_origins
    else None
)

_manager: EngramTideMCPManager | None = None


def get_manager() -> EngramTideMCPManager:
    global _manager
    # FastMCP's server lifespan is scoped to protocol sessions. With stateless
    # HTTP that scope may be one request, while EngramTide conversation state
    # must survive reconnects and successive tool calls. Keep it process-local.
    if _manager is None:
        db_path = os.getenv("ENGRAMTIDE_DB_PATH") or os.getenv("MEMORY_DB_PATH")
        _manager = EngramTideMCPManager(
            EngramTide(db_path=Path(db_path) if db_path else None),
            session_ttl_seconds=_env_int(
                "ENGRAMTIDE_SESSION_TTL_SECONDS", 86_400
            ),
            max_sessions=_env_int("ENGRAMTIDE_MAX_SESSIONS", 32),
        )
    return _manager


def _close_database_at_exit() -> None:
    if _manager is not None:
        _manager.engine.close()


atexit.register(_close_database_at_exit)


mcp = FastMCP(
    "EngramTide",
    instructions=(
        "Long-term memory for one trusted user. Start a session, prepare each "
        "turn before answering, commit the turn after answering, and end the "
        "session when the conversation finishes."
    ),
    host=HOST,
    port=PORT,
    streamable_http_path=PATH,
    stateless_http=True,
    json_response=True,
    transport_security=transport_security,
)


@mcp.tool()
async def start_session(session_id: str | None = None) -> dict[str, Any]:
    """Start or reconnect to one explicit EngramTide conversation session."""
    return await get_manager().start_session(session_id)


@mcp.tool()
async def prepare_turn(
    session_id: str,
    user_input: str,
    turn_id: str | None = None,
    top_k: int | None = None,
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Retrieve memory context before generating the assistant response."""
    return await get_manager().prepare_turn(
        session_id,
        user_input,
        turn_id=turn_id,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
    )


@mcp.tool()
async def commit_turn(
    session_id: str,
    turn_id: str,
    assistant_message: str,
    memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Confirm used memories and record the generated assistant response."""
    return await get_manager().commit_turn(
        session_id, turn_id, assistant_message, memory_ids=memory_ids
    )


@mcp.tool()
async def end_session(session_id: str) -> dict[str, Any]:
    """Extract and write memories, then close a completed conversation session."""
    return await get_manager().end_session(session_id)


@mcp.tool()
async def close_session(session_id: str) -> dict[str, Any]:
    """Close a session without extracting memories; use only when abandoning it."""
    return await get_manager().close_session(session_id)


@mcp.tool()
async def list_memories(limit: int = 100) -> list[dict[str, Any]]:
    """List active memories for inspection, without embedding vectors."""
    return await get_manager().list_memories(limit)


@mcp.tool()
async def export_memories() -> list[dict[str, Any]]:
    """Export all active memories as JSON-compatible records."""
    return await get_manager().export_memories()


@mcp.tool()
async def delete_memories(memory_ids: list[str]) -> dict[str, int]:
    """Permanently delete specified memories and their mechanism logs."""
    return await get_manager().delete_memories(memory_ids)


@mcp.tool()
async def memory_stats() -> dict[str, Any]:
    """Memory-growth observability: counts by type, 30-day net growth, DB size."""
    return await get_manager().memory_stats()


@mcp.tool()
async def search_archive(query_text: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search cold-archived memories. Archiving is not deletion — they are still here."""
    return await get_manager().search_archive(query_text, limit)


@mcp.tool()
async def run_maintenance() -> dict[str, Any]:
    """Run archiving, superseded purge and event-log retention right now."""
    return await get_manager().run_maintenance()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
