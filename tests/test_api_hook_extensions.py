"""Python API 为 hook 服务新增的参数与入口：exclude_ids / query_embedding /
remember / search_memories，以及脱水失败段区间。"""

import uuid

import numpy as np
import pytest

import core.dehydrator as dehydrator
import core.memory_store as memory_store
from core.context_builder import build_constitutional_memory_context
from core.dehydrator import SplitReport, dehydrate_conversation
from core.memory_store import Memory, insert_memory
from engramtide import EngramTide

VEC = np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "ext.db")
    yield
    memory_store.close_db()


@pytest.fixture
def embed_calls(monkeypatch):
    calls: list[str] = []

    async def fake_embed(text: str) -> np.ndarray:
        calls.append(text)
        return VEC

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)
    return calls


def make_memory(memory_type: str, content: str, weight: float = 1.0) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=memory_type,
        content=content,
        decay_weight=weight,
        embedding=VEC,
    )


# ── context_builder.exclude_ids ──────────────────────────


def test_excluded_memories_do_not_use_budget_or_count_as_dropped():
    keep = make_memory("semantic", "用户住在杭州")
    skip = make_memory("semantic", "用户养了一只猫")
    surfaced = make_memory("procedural", "回答先给结论")

    ctx = build_constitutional_memory_context(
        [(keep, 0.9), (skip, 0.8)],
        surfaced=[surfaced],
        exclude_ids={skip.memory_id, surfaced.memory_id},
    )

    assert ctx.included_memory_ids == [keep.memory_id]
    assert ctx.dropped_count == 0
    assert ctx.procedural_memories == "无"


def test_empty_exclude_ids_is_identical_to_default():
    a = make_memory("semantic", "事实 A")
    b = make_memory("episodic", "经历 B")
    retrieved = [(a, 0.9), (b, 0.5)]

    assert build_constitutional_memory_context(retrieved) == (
        build_constitutional_memory_context(retrieved, exclude_ids=frozenset())
    )


# ── prepare_turn 新参数 ───────────────────────────────────


@pytest.mark.asyncio
async def test_precomputed_embedding_skips_embed_call(embed_calls):
    engine = EngramTide(validate_config=False)
    insert_memory(make_memory("semantic", "用户喜欢雨天散步"))
    session = engine.start_session("s")

    await session.prepare_turn("今天下雨了", turn_id="t1", query_embedding=VEC)

    assert embed_calls == []


@pytest.mark.asyncio
async def test_exclude_ids_still_activates_but_does_not_include(embed_calls):
    engine = EngramTide(validate_config=False)
    episodic = make_memory("episodic", "三月被反爬封了 IP", weight=0.08)
    insert_memory(episodic)
    session = engine.start_session("s")

    prepared = await session.prepare_turn(
        "爬虫", turn_id="t1", exclude_ids=[episodic.memory_id]
    )

    assert episodic.memory_id not in prepared.included_memory_ids
    assert episodic.memory_id in prepared.strong_activation_ids
    assert engine.get_memory(episodic.memory_id).decay_weight > 0.08
    assert session.get_prepared("t1") is prepared
    assert session.get_prepared("missing") is None


# ── remember / search_memories ────────────────────────────


@pytest.mark.asyncio
async def test_remember_goes_through_type_aware_writer(embed_calls):
    engine = EngramTide(validate_config=False)

    first = await engine.remember("回答先给结论", "procedural", tags=[" 风格 ", ""])
    duplicate = await engine.remember("回答先给结论", "procedural")

    assert first.inserted == 1
    assert duplicate.inserted == 0 and duplicate.deduped == 1
    [record] = engine.list_memories()
    assert record.tags == ("风格",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": " ", "memory_type": "semantic"},
        {"content": "x", "memory_type": "unknown"},
        {"content": "x", "memory_type": "emotional", "valence": 2.0},
        {"content": "x", "memory_type": "emotional", "arousal": -0.1},
    ],
)
async def test_remember_rejects_invalid_input(embed_calls, kwargs):
    engine = EngramTide(validate_config=False)
    with pytest.raises(ValueError):
        await engine.remember(**kwargs)
    assert embed_calls == []


@pytest.mark.asyncio
async def test_search_memories_is_read_only(embed_calls):
    engine = EngramTide(validate_config=False)
    memory = make_memory("episodic", "三月被反爬封了 IP", weight=0.5)
    insert_memory(memory)

    matches = await engine.search_memories("爬虫", limit=3)

    assert [m.memory.memory_id for m in matches] == [memory.memory_id]
    after = engine.get_memory(memory.memory_id)
    assert after.decay_weight == 0.5
    assert after.access_count == 0


# ── 脱水失败段区间 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_segment_ranges_are_reported(monkeypatch):
    messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]

    async def fake_split(msgs):
        return [msgs[0:2], msgs[2:4], msgs[4:5]], SplitReport(True, 3, False)

    async def fake_segment(seg, _conv_id):
        if seg[0]["content"] == "m2":
            raise RuntimeError("LLM down")
        return []

    monkeypatch.setattr(dehydrator, "split_conversation", fake_split)
    monkeypatch.setattr(dehydrator, "_dehydrate_segment", fake_segment)

    _, report = await dehydrate_conversation(messages, "conv")

    assert report.failed_ranges == [(2, 3)]
