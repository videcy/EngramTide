"""Transport-independent session manager used by the MCP server."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict
from typing import Any, Iterable, Sequence

from engramtide import EngramTide, EngramTideSession


class EngramTideMCPManager:
    """Own one engine and the explicit conversation sessions exposed over MCP."""

    def __init__(
        self,
        engine: EngramTide,
        *,
        session_ttl_seconds: int = 86_400,
        max_sessions: int = 32,
    ) -> None:
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be > 0")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self.engine = engine
        self.session_ttl_seconds = session_ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, EngramTideSession] = {}
        self._last_used: dict[str, float] = {}
        self._committed_turns: set[str] = set()
        self._lock = asyncio.Lock()

    async def start_session(self, session_id: str | None = None) -> dict[str, Any]:
        async with self._lock:
            self._prune_expired()
            resolved = (session_id or str(uuid.uuid4())).strip()
            if not resolved:
                raise ValueError("session_id must be non-empty")
            if resolved in self._sessions:
                self._touch(resolved)
                return {"session_id": resolved, "created": False}
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"maximum active sessions reached ({self.max_sessions}); "
                    "end an existing session first"
                )
            session = self.engine.start_session(resolved)
            self._sessions[resolved] = session
            self._touch(resolved)
            return {
                "session_id": resolved,
                "created": True,
                "surfaced_memory_count": len(session.surfaced_memories),
                "decay_report": asdict(session.decay_report),
            }

    async def prepare_turn(
        self,
        session_id: str,
        user_input: str,
        *,
        turn_id: str | None = None,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            session = self._get_session(session_id)
            kwargs: dict[str, Any] = {"turn_id": turn_id}
            if top_k is not None:
                kwargs["top_k"] = top_k
            if max_context_tokens is not None:
                kwargs["max_context_tokens"] = max_context_tokens
            prepared = await session.prepare_turn(user_input, **kwargs)
            self._touch(session_id)
            return {
                "session_id": session_id,
                "turn_id": prepared.turn_id,
                "memory_context": prepared.prompt_sections(),
                "included_memory_ids": list(prepared.included_memory_ids),
                "retrieval_matches": [
                    {
                        "memory_id": match.memory.memory_id,
                        "type": match.memory.type,
                        "score": match.score,
                        "vector_similarity": match.vector_similarity,
                        "keyword_similarity": match.keyword_similarity,
                    }
                    for match in prepared.retrieval_matches
                ],
            }

    async def commit_turn(
        self,
        session_id: str,
        turn_id: str,
        assistant_message: str,
        *,
        memory_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            session = self._get_session(session_id)
            if not assistant_message.strip():
                raise ValueError("assistant_message must be non-empty")
            receipt = session.acknowledge_used(turn_id, memory_ids)
            # Only append the assistant message on the first successful commit.
            # Replayed commits remain safe and do not duplicate the transcript.
            committed_key = f"{session_id}\0{turn_id}"
            message_recorded = committed_key not in self._committed_turns
            if message_recorded:
                session.add_message("assistant", assistant_message)
                self._committed_turns.add(committed_key)
            self._touch(session_id)
            result = asdict(receipt)
            result["acknowledged_ids"] = list(receipt.acknowledged_ids)
            result["assistant_message_recorded"] = message_recorded
            return result

    async def end_session(
        self,
        session_id: str,
        *,
        conversation: Sequence[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            session = self._get_session(session_id)
            report = await session.end(conversation)
            self._drop_session(session_id)
            return asdict(report)

    async def close_session(self, session_id: str) -> dict[str, Any]:
        """Discard live session state without extracting new memories."""
        async with self._lock:
            session = self._get_session(session_id)
            session.close()
            self._drop_session(session_id)
            return {"session_id": session_id, "closed": True}

    async def list_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        async with self._lock:
            return [record.to_dict() for record in self.engine.list_memories(limit=limit)]

    async def export_memories(self) -> list[dict[str, Any]]:
        async with self._lock:
            return self.engine.export_memories()

    async def delete_memories(self, memory_ids: Iterable[str]) -> dict[str, int]:
        async with self._lock:
            return {"deleted": self.engine.delete_memories(memory_ids)}

    async def shutdown(self) -> None:
        async with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()
            self._last_used.clear()
            self.engine.close()

    def _get_session(self, session_id: str) -> EngramTideSession:
        self._prune_expired()
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown or expired session_id: {session_id}; call start_session first"
            ) from exc

    def _touch(self, session_id: str) -> None:
        self._last_used[session_id] = time.monotonic()

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_used.pop(session_id, None)
        prefix = f"{session_id}\0"
        self._committed_turns = {
            key for key in self._committed_turns if not key.startswith(prefix)
        }

    def _prune_expired(self) -> None:
        cutoff = time.monotonic() - self.session_ttl_seconds
        expired = [sid for sid, touched in self._last_used.items() if touched < cutoff]
        for session_id in expired:
            self._sessions[session_id].close()
            self._drop_session(session_id)
