"""Stable, local Python facade over the EngramTide memory engine.

The existing core modules remain the source of truth for decay, activation,
retrieval and writing.  This module owns the lifecycle and access-counting
contract needed by an external Python agent.

The current storage layer uses a module-level SQLite connection.  Consequently
one Python process should use one EngramTide database at a time.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import config
import core.memory_store as memory_store
from core.context_builder import (
    ConstitutionalMemoryContext,
    build_constitutional_memory_context,
)
from core.decay import ActivationReport, DecayReport, context_aware_update
from core.decay import get_surfaced_memories, run_decay_update
from core.dehydrator import SplitReport, dehydrate_conversation
from core.embedding import embed_text
from core.memory_store import Memory
from core.maintenance import MaintenanceReport, run_maintenance, search_archive
from core.memory_writer import WriteReport, write_memories
from core.retriever import RetrievalDetail, retrieve_memories_detailed
from core.vector_index import reset_cache, similarity_map


@dataclass(frozen=True)
class MemoryRecord:
    """Serializable public representation of a stored memory."""

    memory_id: str
    type: str
    content: str
    valence: float
    arousal: float
    created_at: str
    last_accessed: str
    access_count: int
    decay_weight: float
    source_conv_id: str | None
    unresolved: bool
    tags: tuple[str, ...]
    superseded_by: str | None

    @classmethod
    def from_memory(cls, memory: Memory) -> "MemoryRecord":
        return cls(
            memory_id=memory.memory_id,
            type=memory.type,
            content=memory.content,
            valence=memory.valence,
            arousal=memory.arousal,
            created_at=memory.created_at,
            last_accessed=memory.last_accessed,
            access_count=memory.access_count,
            decay_weight=memory.decay_weight,
            source_conv_id=memory.source_conv_id,
            unresolved=memory.unresolved,
            tags=tuple(memory.tags),
            superseded_by=memory.superseded_by,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


@dataclass(frozen=True)
class RetrievalMatch:
    """Public retrieval score decomposition."""

    memory: MemoryRecord
    score: float
    vector_similarity: float
    keyword_similarity: float | None

    @classmethod
    def from_detail(cls, detail: RetrievalDetail) -> "RetrievalMatch":
        return cls(
            memory=MemoryRecord.from_memory(detail.memory),
            score=detail.score,
            vector_similarity=detail.sim,
            keyword_similarity=detail.kw,
        )


@dataclass(frozen=True)
class PreparedTurn:
    """Memory context prepared for one external-agent turn."""

    turn_id: str
    context: ConstitutionalMemoryContext
    retrieval_matches: tuple[RetrievalMatch, ...]
    strong_activation_ids: tuple[str, ...]
    surfaced_ids: tuple[str, ...]
    activation_report: ActivationReport

    @property
    def included_memory_ids(self) -> tuple[str, ...]:
        return tuple(self.context.included_memory_ids)

    def prompt_sections(self) -> dict[str, str]:
        """Return the four memory sections for a host agent's prompt."""
        return {
            "procedural": self.context.procedural_memories,
            "semantic": self.context.semantic_memories,
            "episodic": self.context.episodic_memories,
            "emotional": self.context.emotional_memories,
        }


@dataclass(frozen=True)
class AccessReceipt:
    """Idempotent acknowledgement result for one prepared turn."""

    turn_id: str
    acknowledged_ids: tuple[str, ...]
    surface_events: int
    retrieval_events: int
    already_acknowledged: int


@dataclass(frozen=True)
class SessionEndReport:
    """Summary returned after dehydration and differentiated writing."""

    session_id: str
    extracted_memories: int
    split_report: SplitReport
    write_report: WriteReport


class EngramTide:
    """Local, single-database Python API for EngramTide.

    Args:
        db_path: Optional SQLite path. Setting it switches the process-wide
            EngramTide connection to that database.
        validate_config: Validate chat and embedding provider settings. Disable
            only for offline inspection/tests that never call network features.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        validate_config: bool = True,
    ) -> None:
        if validate_config:
            errors = config.check_config()
            if errors:
                raise ValueError("EngramTide configuration is invalid:\n" + "\n".join(errors))

        if db_path is not None:
            target = Path(db_path).expanduser().resolve()
            if Path(memory_store.DB_PATH).resolve() != target:
                memory_store.close_db()
                memory_store.DB_PATH = target
                # 换库必须丢掉进程内的向量矩阵，否则会拿旧库的记忆去打分
                reset_cache()

        memory_store.init_db()
        self.db_path = Path(memory_store.DB_PATH).resolve()

    def start_session(
        self,
        session_id: str | None = None,
        *,
        now: datetime | None = None,
    ) -> "EngramTideSession":
        """Run session-start decay and collect (but do not count) surfaced memories.

        Also runs frequency-gated maintenance (event-log retention, superseded
        purge, cold archiving) so a long-lived host process still gets an exit
        path for memory growth.
        """
        decay_report = run_decay_update(now=now)
        maintenance_report = run_maintenance(now=now)
        surfaced = get_surfaced_memories(memory_store.list_active_memories(), now=now)
        return EngramTideSession(
            engine=self,
            session_id=session_id or str(uuid.uuid4()),
            decay_report=decay_report,
            surfaced_memories=surfaced,
            maintenance_report=maintenance_report,
        )

    def stats(self) -> dict[str, Any]:
        """Memory-growth observability: counts, 30-day net growth, DB size."""
        return memory_store.memory_stats()

    def search_archive(self, query_text: str, limit: int = 10) -> list[MemoryRecord]:
        """Search cold-archived memories. Archiving is not deletion."""
        return [
            MemoryRecord.from_memory(m) for m in search_archive(query_text, limit)
        ]

    def restore_archived(self, memory_ids: Iterable[str]) -> int:
        """Move archived memories back into the hot table."""
        return memory_store.restore_archived(list(memory_ids))

    def run_maintenance(self, *, force: bool = True) -> MaintenanceReport:
        """Run archiving / retention maintenance on demand."""
        return run_maintenance(force=force)

    def list_memories(self, *, limit: int | None = None) -> list[MemoryRecord]:
        memories = memory_store.list_active_memories()
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be >= 0")
            memories = memories[:limit]
        return [MemoryRecord.from_memory(memory) for memory in memories]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        memory = memory_store.get_memory(memory_id)
        return MemoryRecord.from_memory(memory) if memory else None

    def delete_memories(self, memory_ids: Iterable[str]) -> int:
        """Hard-delete memories and their mechanism logs."""
        return memory_store.delete_memories(list(memory_ids))

    def export_memories(self) -> list[dict[str, Any]]:
        """Return active memories as JSON-serializable dictionaries (no embeddings)."""
        return [record.to_dict() for record in self.list_memories()]

    def close(self) -> None:
        memory_store.close_db()

    def __enter__(self) -> "EngramTide":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class EngramTideSession:
    """Stateful local conversation lifecycle for a host Python agent."""

    def __init__(
        self,
        *,
        engine: EngramTide,
        session_id: str,
        decay_report: DecayReport,
        surfaced_memories: Sequence[Memory],
        maintenance_report: MaintenanceReport | None = None,
    ) -> None:
        self.engine = engine
        self.session_id = session_id
        self.decay_report = decay_report
        self.maintenance_report = maintenance_report or MaintenanceReport()
        self.surfaced_memories = tuple(surfaced_memories)
        self._surfaced_ids = {memory.memory_id for memory in surfaced_memories}
        self._surface_acknowledged_ids: set[str] = set()
        self._mild_ids: set[str] = set()
        self._prepared: dict[str, PreparedTurn] = {}
        self._acknowledged_by_turn: dict[str, set[str]] = {}
        self._conversation: list[dict[str, str]] = []
        self._closed = False

    @property
    def conversation(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(message) for message in self._conversation)

    def add_message(self, role: str, content: str) -> None:
        self._ensure_open()
        role = role.strip()
        content = content.strip()
        if not role or not content:
            raise ValueError("role and content must be non-empty")
        self._conversation.append({"role": role, "content": content})

    async def prepare_turn(
        self,
        user_input: str,
        *,
        turn_id: str | None = None,
        top_k: int = config.TOP_K_RETRIEVE,
        max_context_tokens: int = config.MAX_CONTEXT_TOKENS,
        debug: bool = False,
        record_user_message: bool = True,
    ) -> PreparedTurn:
        """Prepare memory context without counting retrieval/surface use.

        Strong activation keeps the existing core semantics: it is an input-
        triggered mechanism event and is counted during this method.  Retrieval
        and surfacing are counted only by acknowledge_used().
        """
        self._ensure_open()
        text = user_input.strip()
        if not text:
            raise ValueError("user_input must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        resolved_turn_id = turn_id or str(uuid.uuid4())
        if resolved_turn_id in self._prepared:
            raise ValueError(f"turn_id already prepared: {resolved_turn_id}")

        query_embedding = await embed_text(text)

        # One load + one similarity matmul per turn, shared by activation and
        # retrieval.  context_aware_update() writes the new decay weights back
        # into these Memory objects, so retrieval below scores against the
        # post-activation weights exactly as it did when it re-read the DB.
        turn_memories = memory_store.list_active_memories()
        turn_sims = similarity_map(query_embedding, turn_memories)

        activation_report = context_aware_update(
            query_embedding,
            memories=turn_memories,
            exclude_mild_ids=(
                frozenset(self._mild_ids)
                if config.MILD_ONCE_PER_SESSION
                else frozenset()
            ),
            sims=turn_sims,
        )
        if config.MILD_ONCE_PER_SESSION:
            self._mild_ids.update(activation_report.mild_ids)

        details = retrieve_memories_detailed(
            query_embedding,
            memories=turn_memories,
            top_k=top_k,
            query_text=text,
            sims=turn_sims,
        )
        retrieved = [(detail.memory, detail.score) for detail in details]
        context = build_constitutional_memory_context(
            retrieved,
            max_tokens=max_context_tokens,
            debug=debug,
            surfaced=list(self.surfaced_memories),
        )
        prepared = PreparedTurn(
            turn_id=resolved_turn_id,
            context=context,
            retrieval_matches=tuple(RetrievalMatch.from_detail(d) for d in details),
            strong_activation_ids=tuple(activation_report.strong_ids),
            surfaced_ids=tuple(self._surfaced_ids),
            activation_report=activation_report,
        )
        self._prepared[resolved_turn_id] = prepared
        self._acknowledged_by_turn[resolved_turn_id] = set()

        if record_user_message:
            self.add_message("user", text)
        return prepared

    def acknowledge_used(
        self,
        turn_id: str,
        memory_ids: Iterable[str] | None = None,
    ) -> AccessReceipt:
        """Count memories actually included by the host agent, idempotently."""
        self._ensure_open()
        if turn_id not in self._prepared:
            raise KeyError(f"unknown turn_id: {turn_id}")

        prepared = self._prepared[turn_id]
        allowed_ids = set(prepared.included_memory_ids)
        requested = allowed_ids if memory_ids is None else set(memory_ids)
        unknown = requested - allowed_ids
        if unknown:
            raise ValueError(
                "memory_ids were not included in this prepared turn: "
                + ", ".join(sorted(unknown))
            )

        acknowledged = self._acknowledged_by_turn[turn_id]
        duplicate_count = len(requested & acknowledged)
        new_ids = requested - acknowledged
        acknowledged.update(new_ids)

        strong_ids = set(prepared.strong_activation_ids)
        surface_ids = (
            new_ids & self._surfaced_ids
        ) - self._surface_acknowledged_ids
        retrieval_ids = new_ids - self._surfaced_ids - strong_ids

        self._record_access(surface_ids, "surface", config.ABL_ACCESS_SURFACE)
        self._record_access(retrieval_ids, "retrieval", config.ABL_ACCESS_RETRIEVAL)
        self._surface_acknowledged_ids.update(surface_ids)

        return AccessReceipt(
            turn_id=turn_id,
            acknowledged_ids=tuple(sorted(new_ids)),
            surface_events=len(surface_ids),
            retrieval_events=len(retrieval_ids),
            already_acknowledged=duplicate_count,
        )

    async def end(
        self,
        conversation: Sequence[dict[str, str]] | None = None,
    ) -> SessionEndReport:
        """Dehydrate the conversation and run differentiated memory writing."""
        self._ensure_open()
        messages = list(conversation) if conversation is not None else list(self._conversation)
        memories, split_report = await dehydrate_conversation(messages, self.session_id)
        write_report = await write_memories(memories)
        self._closed = True
        return SessionEndReport(
            session_id=self.session_id,
            extracted_memories=len(memories),
            split_report=split_report,
            write_report=write_report,
        )

    def close(self) -> None:
        """Close the logical session without extracting conversation memories."""
        self._closed = True

    def _record_access(self, memory_ids: set[str], source: str, enabled: bool) -> None:
        if not memory_ids:
            return
        ordered = sorted(memory_ids)
        if enabled:
            memory_store.mark_accessed(ordered)
        if config.ACCESS_LOG_ENABLED:
            logged_source = source if enabled else f"{source}_ablated"
            memory_store.log_access_events_batch(
                [(memory_id, logged_source) for memory_id in ordered]
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EngramTide session is closed")
