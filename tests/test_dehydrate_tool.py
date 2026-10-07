"""MemoryService 写路径：dehydrate 增量/补扫/幂等/部分失败，及其余 MCP 写工具。"""

import itertools
import uuid

import numpy as np
import pytest

import core.memory_store as memory_store
from core.dehydrator import SplitReport
from core.memory_store import Memory, insert_memory, list_pending_turns, upsert_turn
from engramtide import EngramTide
from engramtide.service import MemoryService

VEC = np.array([1.0, 0.0], dtype=np.float32)
_axis = itertools.count()


def _distinct_vec() -> np.ndarray:
    """互相正交的向量，避免 episodic 近重复去重把测试记忆合并掉。"""
    vec = np.zeros(64, dtype=np.float32)
    vec[next(_axis) % 64] = 1.0
    return vec


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "dehydrate.db")
    yield
    memory_store.close_db()


@pytest.fixture
def service(monkeypatch):
    async def fake_embed(_: str) -> np.ndarray:
        return VEC

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)
    return MemoryService(EngramTide(validate_config=False), template="{session_id}{sections}")


class FakeDehydrator:
    """每条用户消息产出一条 episodic；fail_ranges 内的消息视为失败段。"""

    def __init__(self, fail_ranges=(), raise_all=False):
        self.calls: list[list[str]] = []
        self.fail_ranges = list(fail_ranges)
        self.raise_all = raise_all

    async def __call__(self, messages, source_conv_id):
        self.calls.append([m["content"] for m in messages])
        if self.raise_all:
            raise RuntimeError("LLM down")
        failed = {i for s, e in self.fail_ranges for i in range(s, e + 1)}
        memories = [
            Memory(
                memory_id=str(uuid.uuid4()),
                type="episodic",
                content=f"{source_conv_id}:{m['content']}",
                embedding=_distinct_vec(),
                source_conv_id=source_conv_id,
            )
            for i, m in enumerate(messages)
            if m["role"] == "user" and i not in failed
        ]
        report = SplitReport(True, 1, False, failed_ranges=list(self.fail_ranges))
        return memories, report


def install(monkeypatch, fake):
    monkeypatch.setattr("engramtide.service.dehydrate_conversation", fake)
    return fake


def record(session_id, n, *, answered=True, start=0):
    for i in range(start, start + n):
        upsert_turn(session_id, f"p{i}", "user", f"问{i}")
        if answered or i < start + n - 1:
            upsert_turn(session_id, f"p{i}", "assistant", f"答{i}")


@pytest.mark.asyncio
async def test_incremental_and_idempotent(service, monkeypatch):
    fake = install(monkeypatch, FakeDehydrator())
    record("s1", 2)

    first = await service.dehydrate("s1")
    again = await service.dehydrate("s1")
    record("s1", 1, start=2)
    third = await service.dehydrate("s1")

    [r1] = first["sessions"]
    assert (r1["messages"], r1["marked_messages"], r1["write_report"]["inserted"]) == (4, 4, 2)
    assert again["sessions"][0]["skipped"] == "no pending turns"
    assert fake.calls == [["问0", "答0", "问1", "答1"], ["问2", "答2"]]
    assert third["extracted_memories"] == 1
    assert len(service.engine.list_memories()) == 3


@pytest.mark.asyncio
async def test_dry_run_neither_writes_nor_marks(service, monkeypatch):
    install(monkeypatch, FakeDehydrator())
    record("s1", 1)

    result = await service.dehydrate("s1", dry_run=True)

    assert result["dry_run"] is True
    assert result["sessions"][0]["memories"] == [{"type": "episodic", "content": "s1:问0"}]
    assert service.engine.list_memories() == []
    assert len(list_pending_turns("s1")) == 2


@pytest.mark.asyncio
async def test_partial_failure_marks_only_successful_segments(service, monkeypatch):
    install(monkeypatch, FakeDehydrator(fail_ranges=[(2, 3)]))
    record("s1", 3)

    [r] = (await service.dehydrate("s1"))["sessions"]

    assert (r["marked_messages"], r["pending_messages"]) == (4, 2)
    assert [row["content"] for row in list_pending_turns("s1")] == ["问1", "答1"]


@pytest.mark.asyncio
async def test_total_failure_marks_nothing(service, monkeypatch):
    install(monkeypatch, FakeDehydrator(raise_all=True))
    record("s1", 2)

    [r] = (await service.dehydrate("s1"))["sessions"]

    assert r["error"].startswith("dehydration failed")
    assert len(list_pending_turns("s1")) == 4


@pytest.mark.asyncio
async def test_session_resolution(service, monkeypatch):
    install(monkeypatch, FakeDehydrator())

    assert (await service.dehydrate())["sessions"] == []

    record("only", 1)
    assert (await service.dehydrate())["sessions"][0]["session_id"] == "only"

    record("a", 1)
    record("b", 1)
    with pytest.raises(ValueError, match="a \\(2\\)"):
        await service.dehydrate()
    with pytest.raises(ValueError):
        await service.dehydrate("a", scope="all")


@pytest.mark.asyncio
async def test_pending_scope_sweeps_every_session(service, monkeypatch):
    install(monkeypatch, FakeDehydrator())
    record("a", 1)
    record("b", 2)

    result = await service.dehydrate(scope="pending")

    assert {r["session_id"] for r in result["sessions"]} == {"a", "b"}
    assert result["extracted_memories"] == 3
    assert list_pending_turns() == []


@pytest.mark.asyncio
async def test_in_progress_turn_of_live_session_is_left_for_later(service, monkeypatch):
    fake = install(monkeypatch, FakeDehydrator())
    await service.on_user_prompt("live", "第一个问题", prompt_id="p0")
    await service.on_stop("live", "第一个回答", prompt_id="p0")
    await service.on_user_prompt("live", "请记住刚才的讨论", prompt_id="p1")

    await service.dehydrate("live")
    record("dead", 1, answered=False)  # 不在内存里的会话：未回复的那轮也处理
    await service.dehydrate("dead")

    assert fake.calls == [["第一个问题", "第一个回答"], ["问0"]]
    assert [r["content"] for r in list_pending_turns("live")] == ["请记住刚才的讨论"]


@pytest.mark.asyncio
async def test_remember_forget_restore_and_stats(service):
    report = await service.remember("用户住在杭州", "semantic", tags=["城市"], session_id="s1")
    assert report["inserted"] == 1
    [record_] = service.list_memories()
    assert record_["source_conv_id"] == "s1"

    upsert_turn("s9", "p0", "user", "未脱水")
    stats = service.stats()
    assert stats["pending_turns"] == 1
    assert stats["pending_sessions"][0]["session_id"] == "s9"

    assert service.forget([record_["memory_id"]]) == {"deleted": 1}
    assert service.restore_archived(["missing"]) == {"restored": 0}


@pytest.mark.asyncio
async def test_search_memories_returns_scores(service):
    insert_memory(Memory(
        memory_id="m1", type="semantic", content="用户住在杭州", embedding=VEC,
    ))

    [hit] = await service.search_memories("杭州", limit=5)

    assert hit["memory_id"] == "m1"
    assert hit["score"] > 0
    assert "embedding" not in hit


@pytest.mark.asyncio
async def test_consolidate_preview_lists_pairs_without_llm(service):
    for memory_id in ("e1", "e2"):
        insert_memory(Memory(
            memory_id=memory_id, type="episodic", content=f"去了西湖 {memory_id}",
            embedding=VEC,
        ))

    result = await service.consolidate(preview=True)

    assert result["preview"] is True
    [pair] = result["pairs"]
    assert {m["memory_id"] for m in pair["memories"]} == {"e1", "e2"}
