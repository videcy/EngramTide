"""Public Python API contract tests."""

import uuid

import numpy as np
import pytest

import core.memory_store as memory_store
from core.memory_store import Memory, insert_memory, query_access_events
from engramtide import EngramTide


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "api.db")
    yield
    memory_store.close_db()


@pytest.fixture
def fake_query_embedding(monkeypatch):
    async def fake_embed(_: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)


def make_memory(memory_type: str, content: str) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=memory_type,
        content=content,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_prepare_then_acknowledge_retrieval_is_idempotent(fake_query_embedding):
    engine = EngramTide(validate_config=False)
    memory = make_memory("semantic", "用户喜欢雨天散步")
    insert_memory(memory)

    session = engine.start_session("conversation-1")
    prepared = await session.prepare_turn("今天下雨了", turn_id="turn-1")

    assert memory.memory_id in prepared.included_memory_ids
    assert engine.get_memory(memory.memory_id).access_count == 0
    assert prepared.prompt_sections()["semantic"] != "无"

    receipt = session.acknowledge_used("turn-1")
    assert receipt.retrieval_events == 1
    assert engine.get_memory(memory.memory_id).access_count == 1

    duplicate = session.acknowledge_used("turn-1")
    assert duplicate.already_acknowledged == 1
    assert duplicate.retrieval_events == 0
    assert engine.get_memory(memory.memory_id).access_count == 1


@pytest.mark.asyncio
async def test_surface_counts_only_after_actual_use_and_once_per_session(fake_query_embedding):
    engine = EngramTide(validate_config=False)
    memory = make_memory("procedural", "回复前先确认对方是否需要建议")
    insert_memory(memory)

    session = engine.start_session("conversation-2")
    assert memory.memory_id in {m.memory_id for m in session.surfaced_memories}
    assert engine.get_memory(memory.memory_id).access_count == 0

    first = await session.prepare_turn("我今天很累", turn_id="turn-1")
    assert memory.memory_id in first.included_memory_ids
    first_receipt = session.acknowledge_used("turn-1")
    assert first_receipt.surface_events == 1
    assert engine.get_memory(memory.memory_id).access_count == 1

    await session.prepare_turn("还是不太舒服", turn_id="turn-2")
    second_receipt = session.acknowledge_used("turn-2")
    assert second_receipt.surface_events == 0
    assert second_receipt.retrieval_events == 0
    assert engine.get_memory(memory.memory_id).access_count == 1


@pytest.mark.asyncio
async def test_acknowledge_rejects_memory_not_in_prepared_context(fake_query_embedding):
    engine = EngramTide(validate_config=False)
    session = engine.start_session()
    await session.prepare_turn("测试", turn_id="turn-1")

    with pytest.raises(ValueError, match="not included"):
        session.acknowledge_used("turn-1", ["unknown-memory"])


def test_export_and_hard_delete_remove_logs():
    engine = EngramTide(validate_config=False)
    memory = make_memory("semantic", "用户偏好简洁回答")
    memory.tags = ["preference"]
    insert_memory(memory)
    memory_store.mark_accessed([memory.memory_id])
    memory_store.log_access_event(memory.memory_id, "retrieval")

    exported = engine.export_memories()
    assert exported[0]["memory_id"] == memory.memory_id
    assert exported[0]["tags"] == ["preference"]
    assert "embedding" not in exported[0]

    assert engine.delete_memories([memory.memory_id]) == 1
    assert engine.get_memory(memory.memory_id) is None
    assert query_access_events(memory.memory_id) == []


@pytest.mark.asyncio
async def test_end_empty_session_and_closed_contract():
    engine = EngramTide(validate_config=False)
    session = engine.start_session("conversation-empty")

    report = await session.end([])
    assert report.extracted_memories == 0
    assert report.write_report.inserted == 0

    with pytest.raises(RuntimeError, match="closed"):
        session.add_message("user", "hello")
