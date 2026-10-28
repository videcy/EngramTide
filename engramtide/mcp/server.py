"""EngramTide over Streamable HTTP: MCP write/admin tools + Claude Code hooks.

One process serves both, so hooks and tools share one SQLite connection, one
vector cache and one set of live sessions:

* ``/mcp`` — MCP tools.  Writing memories (dehydrate, remember, consolidate,
  forget …) happens only here, only when explicitly called.
* ``/hooks/claude-code/*`` — HTTP hooks.  UserPromptSubmit injects memory
  every turn; Stop and PostCompact keep the conversation buffer and the
  injection dedup in step.  Hooks never write memories.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from engramtide import EngramTide
from engramtide.hooks.claude_code import build_hook_handlers
from engramtide.service import MemoryService


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
HOOK_TOKEN = os.getenv("ENGRAMTIDE_HOOK_TOKEN", "").strip() or None


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

_service: MemoryService | None = None


def get_service() -> MemoryService:
    global _service
    # FastMCP's server lifespan is scoped to protocol sessions. With stateless
    # HTTP that scope may be one request, while hook session state must
    # survive across requests. Keep it process-local.
    if _service is None:
        db_path = os.getenv("ENGRAMTIDE_DB_PATH") or os.getenv("MEMORY_DB_PATH")
        _service = MemoryService(
            EngramTide(db_path=Path(db_path) if db_path else None),
            session_ttl_seconds=_env_int("ENGRAMTIDE_SESSION_TTL_SECONDS", 86_400),
            max_sessions=_env_int("ENGRAMTIDE_MAX_SESSIONS", 32),
        )
    return _service


def _close_database_at_exit() -> None:
    if _service is not None:
        _service.engine.close()


atexit.register(_close_database_at_exit)


mcp = FastMCP(
    "EngramTide",
    instructions=(
        "Long-term memory for one trusted user. Relevant memories are injected "
        "automatically before each user prompt inside an <engramtide-memory> "
        "block; do not call tools to fetch them. Use search_memories only for an "
        "explicit lookup. When a task or discussion reaches a natural end, or "
        "the user asks you to remember the conversation, call dehydrate with the "
        "session id from the <engramtide-memory> tag. Use remember for a single "
        "fact the user explicitly wants kept. forget is permanent: confirm with "
        "the user first."
    ),
    host=HOST,
    port=PORT,
    streamable_http_path=PATH,
    stateless_http=True,
    json_response=True,
    transport_security=transport_security,
)

for _path, _handler in build_hook_handlers(
    get_service,
    security=mcp.settings.transport_security,
    token=HOOK_TOKEN,
).items():
    mcp.custom_route(_path, methods=["POST"], include_in_schema=False)(_handler)


# ── 写入 ─────────────────────────────────────────────────


@mcp.tool()
async def dehydrate(
    session_id: str | None = None,
    scope: Literal["session", "pending"] = "session",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract memories from the recorded conversation and write them.

    Only turns not yet dehydrated are processed, so calling it again is safe.
    scope="session" handles one session (session_id from the <engramtide-memory>
    tag; may be omitted when only one session has pending turns).
    scope="pending" sweeps every session with pending turns.
    dry_run=True returns what would be extracted without writing.
    """
    return await get_service().dehydrate(session_id, scope=scope, dry_run=dry_run)


@mcp.tool()
async def remember(
    content: str,
    type: Literal["semantic", "episodic", "emotional", "procedural"],
    valence: float = 0.0,
    arousal: float = 0.0,
    unresolved: bool = False,
    tags: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Write one memory the user explicitly wants kept.

    semantic = facts about the user; episodic = events; emotional = experiences
    with feelings (valence -1..1, arousal 0..1); procedural = preferences and
    rules for how to work with the user. Override, reinforcement and dedup
    rules apply, so the report may show no new insertion.
    """
    return await get_service().remember(
        content,
        type,
        valence=valence,
        arousal=arousal,
        unresolved=unresolved,
        tags=tags or (),
        session_id=session_id,
    )


@mcp.tool()
async def forget(memory_ids: list[str]) -> dict[str, int]:
    """Permanently delete memories and their logs. Confirm with the user first."""
    return get_service().forget(memory_ids)


@mcp.tool()
async def consolidate(preview: bool = True) -> dict[str, Any]:
    """Merge near-duplicate memories. preview=True lists candidate pairs at zero cost."""
    return await get_service().consolidate(preview)


@mcp.tool()
async def restore_archived(memory_ids: list[str]) -> dict[str, int]:
    """Move archived memories back into active memory."""
    return get_service().restore_archived(memory_ids)


@mcp.tool()
async def run_maintenance() -> dict[str, Any]:
    """Run archiving, superseded purge, event-log and buffer retention right now."""
    return get_service().run_maintenance()


# ── 只读 ─────────────────────────────────────────────────


@mcp.tool()
async def search_memories(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Explicitly look up memories. Read-only: changes no weights or access counts."""
    return await get_service().search_memories(query, limit)


@mcp.tool()
async def list_memories(limit: int = 100) -> list[dict[str, Any]]:
    """List active memories for inspection, without embedding vectors."""
    return get_service().list_memories(limit)


@mcp.tool()
async def export_memories() -> list[dict[str, Any]]:
    """Export all active memories as JSON-compatible records."""
    return get_service().export_memories()


@mcp.tool()
async def memory_stats() -> dict[str, Any]:
    """Counts by type, 30-day net growth, DB size, and sessions awaiting dehydration."""
    return get_service().stats()


@mcp.tool()
async def search_archive(query_text: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search cold-archived memories. Archiving is not deletion — they are still here."""
    return get_service().search_archive(query_text, limit)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
