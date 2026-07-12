"""Tests for the transport-independent MCP lifecycle adapter."""

import uuid

import numpy as np
import pytest

import core.memory_store as memory_store
from core.memory_store import Memory, insert_memory
from engramtide import EngramTide
from engramtide.mcp import EngramTideMCPManager


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "mcp.db")
    yield
    memory_store.close_db()


@pytest.fixture
def fake_embedding(monkeypatch):
    async def fake_embed(_: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)


def make_memory() -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type="semantic",
        content="用户喜欢在雨天散步",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_explicit_session_prepare_and_commit_are_replay_safe(fake_embedding):
    engine = EngramTide(validate_config=False)
    manager = EngramTideMCPManager(engine)
    memory = make_memory()
    insert_memory(memory)

    created = await manager.start_session("chat-1")
    reconnected = await manager.start_session("chat-1")
    prepared = await manager.prepare_turn(
        "chat-1", "今天下雨", turn_id="turn-1"
    )
    first = await manager.commit_turn(
        "chat-1", "turn-1", "要不要去散散步？"
    )
    replay = await manager.commit_turn(
        "chat-1", "turn-1", "这条回复不应重复写入"
    )

    assert created["created"] is True
    assert reconnected == {"session_id": "chat-1", "created": False}
    assert memory.memory_id in prepared["included_memory_ids"]
    assert first["assistant_message_recorded"] is True
    assert replay["assistant_message_recorded"] is False
    assert engine.get_memory(memory.memory_id).access_count == 1
    assert manager._sessions["chat-1"].conversation == (
        {"role": "user", "content": "今天下雨"},
        {"role": "assistant", "content": "要不要去散散步？"},
    )


@pytest.mark.asyncio
async def test_capacity_close_and_unknown_session_contract():
    manager = EngramTideMCPManager(
        EngramTide(validate_config=False), max_sessions=1
    )
    await manager.start_session("one")

    with pytest.raises(RuntimeError, match="maximum active sessions"):
        await manager.start_session("two")

    assert await manager.close_session("one") == {
        "session_id": "one",
        "closed": True,
    }
    with pytest.raises(KeyError, match="unknown or expired"):
        await manager.prepare_turn("one", "hello")


@pytest.mark.asyncio
async def test_empty_assistant_message_does_not_acknowledge(fake_embedding):
    engine = EngramTide(validate_config=False)
    manager = EngramTideMCPManager(engine)
    memory = make_memory()
    insert_memory(memory)
    await manager.start_session("chat")
    await manager.prepare_turn("chat", "雨天", turn_id="turn")

    with pytest.raises(ValueError, match="assistant_message"):
        await manager.commit_turn("chat", "turn", "  ")

    assert engine.get_memory(memory.memory_id).access_count == 0
