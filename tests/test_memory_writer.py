"""
测试写入管线：四类写入策略（semantic 覆盖 / emotional 强化 / procedural 去重 / episodic 直写）。
"""

import uuid

import numpy as np
import pytest

from core.memory_store import (
    Memory,
    close_db,
    init_db,
    insert_memory,
    list_active_memories,
)
from core.memory_writer import write_memories, WriteReport


def _make_mem(
    mem_type: str,
    content: str,
    embedding: np.ndarray | None = None,
    valence: float = 0.0,
    arousal: float = 0.0,
    tags: list[str] | None = None,
    decay_weight: float = 1.0,
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=embedding,
        valence=valence,
        arousal=arousal,
        tags=tags or [],
        decay_weight=decay_weight,
    )


# 确定性的 fake embedding：内容不同 → 向量不同（用 hash 种子）
def _fake_embed(text: str) -> np.ndarray:
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    vec = rng.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """独立临时数据库。"""
    db_file = tmp_path / "test_writer.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)

    import core.memory_store as ms

    ms._connection = None
    init_db()
    yield
    close_db()


# ── Episodic：直接写入 ─────────────────────────────────────


class TestEpisodicWrite:
    @pytest.mark.asyncio
    async def test_episodic_direct_insert(self):
        mem = _make_mem("episodic", "昨天吃了火锅", embedding=_fake_embed("昨天吃了火锅"))
        report = await write_memories([mem])
        assert report.inserted == 1
        assert report.superseded == 0

        active = list_active_memories()
        assert len(active) == 1
        assert active[0].content == "昨天吃了火锅"

    @pytest.mark.asyncio
    async def test_episodic_no_dedup_on_duplicate(self):
        """episodic 不做去重，同样内容也会插入两次。"""
        emb = _fake_embed("事件")
        m1 = _make_mem("episodic", "事件 A", embedding=emb)
        m2 = _make_mem("episodic", "事件 A", embedding=emb)

        await write_memories([m1])
        report = await write_memories([m2])
        assert report.inserted == 1

        active = list_active_memories()
        assert len(active) == 2


# ── Semantic：覆盖检测 ─────────────────────────────────────


class TestSemanticWrite:
    @pytest.mark.asyncio
    async def test_semantic_direct_insert_when_no_match(self):
        mem = _make_mem("semantic", "用户住在东京", embedding=_fake_embed("用户住在东京"))
        report = await write_memories([mem])
        assert report.inserted == 1
        assert report.superseded == 0

    @pytest.mark.asyncio
    async def test_semantic_override_on_high_similarity(self):
        """高相似度 semantic → 旧记忆被标记 superseded。"""
        # 先插入旧语义记忆
        old = _make_mem("semantic", "用户住在东京", embedding=_fake_embed("用户住在东京"))
        insert_memory(old)

        # 新记忆：搬家
        new = _make_mem("semantic", "用户搬到了大阪", embedding=_fake_embed("用户搬到了大阪"))
        report = await write_memories([new])
        assert report.inserted == 1
        # 注意：fake_embed 基于内容 hash，"用户住在东京" 和 "用户搬到了大阪" 
        # 字符不完全相同，相似度可能不达阈值。
        # 这里只验证管线不崩溃，覆盖逻辑用更精确的测试。

    @pytest.mark.asyncio
    async def test_semantic_override_with_identical_embedding(self):
        """相同 embedding → 必定触发覆盖（相似度 1.0）。"""
        emb = _fake_embed("相同内容")
        old = _make_mem("semantic", "相同内容旧版", embedding=emb)
        insert_memory(old)

        new = _make_mem("semantic", "相同内容新版", embedding=emb)
        report = await write_memories([new])
        assert report.inserted == 1
        assert report.superseded == 1

        # 旧记忆被标记 superseded
        active = list_active_memories()
        assert len(active) == 1
        assert active[0].memory_id == new.memory_id

    @pytest.mark.asyncio
    async def test_semantic_no_override_on_low_similarity(self):
        """低相似度 → 两条都在。"""
        old = _make_mem("semantic", "用户喜欢看电影", embedding=_fake_embed("用户喜欢看电影"))
        insert_memory(old)

        new = _make_mem("semantic", "用户住在东京", embedding=_fake_embed("用户住在东京"))
        report = await write_memories([new])
        assert report.inserted == 1
        assert report.superseded == 0

        active = list_active_memories()
        assert len(active) == 2


# ── Emotional：强化写入 ────────────────────────────────────


class TestEmotionalWrite:
    @pytest.mark.asyncio
    async def test_emotional_direct_insert_when_no_match(self):
        mem = _make_mem("emotional", "用户厌恶机械回复", embedding=_fake_embed("用户厌恶机械回复"))
        report = await write_memories([mem])
        assert report.inserted == 1
        assert report.reinforced == 0

    @pytest.mark.asyncio
    async def test_emotional_reinforce_on_identical_embedding(self):
        """相同 embedding → 强化旧记忆，不 insert。"""
        emb = _fake_embed("相同情绪")
        old = _make_mem(
            "emotional",
            "用户讨厌画大饼",
            embedding=emb,
            arousal=0.5,
            decay_weight=0.6,
            tags=["职场"],
        )
        insert_memory(old)

        new = _make_mem(
            "emotional",
            "用户讨厌画大饼（再次吐槽）",
            embedding=emb,
            arousal=0.8,
            tags=["吐槽"],
        )
        report = await write_memories([new])
        assert report.inserted == 0
        assert report.reinforced == 1

        # 验证强化结果：decay_weight 提升、arousal 取 max、tags 合并
        active = list_active_memories()
        assert len(active) == 1
        updated = active[0]
        assert updated.decay_weight == pytest.approx(0.8)  # 0.6 + 0.2
        assert updated.arousal == 0.8  # max(0.5, 0.8)
        assert set(updated.tags) == {"职场", "吐槽"}  # 合并去重
        assert updated.access_count == 1  # access_count +1


# ── Procedural：写入去重 ───────────────────────────────────


class TestProceduralWrite:
    @pytest.mark.asyncio
    async def test_procedural_direct_insert_when_no_match(self):
        mem = _make_mem(
            "procedural",
            "用户吐槽时先共情",
            embedding=_fake_embed("用户吐槽时先共情"),
        )
        report = await write_memories([mem])
        assert report.inserted == 1
        assert report.deduped == 0

    @pytest.mark.asyncio
    async def test_procedural_dedup_on_identical_embedding(self):
        """相同 embedding → 去重，不 insert，mark_accessed 旧记忆。"""
        emb = _fake_embed("相同规则")
        old = _make_mem("procedural", "先共情再讲道理", embedding=emb)
        insert_memory(old)

        assert list_active_memories()[0].access_count == 0

        new = _make_mem("procedural", "先共情再讲道理（重复）", embedding=emb)
        report = await write_memories([new])
        assert report.inserted == 0
        assert report.deduped == 1

        # 只有一个 procedural
        active = list_active_memories()
        assert len(active) == 1
        assert active[0].access_count == 1  # mark_accessed


# ── 错误处理与边界 ─────────────────────────────────────────


class TestWriteErrorHandling:
    @pytest.mark.asyncio
    async def test_empty_list_returns_zero_report(self):
        report = await write_memories([])
        assert report == WriteReport()

    @pytest.mark.asyncio
    async def test_unknown_type_direct_insert(self):
        """未知类型降级为直接 insert。"""
        mem = _make_mem("unknown_type", "未知", embedding=_fake_embed("未知"))
        report = await write_memories([mem])
        assert report.inserted == 1

    @pytest.mark.asyncio
    async def test_null_embedding_direct_insert(self):
        """无 embedding 降级为直接 insert。"""
        mem = _make_mem("semantic", "无 embedding 的记忆", embedding=None)
        report = await write_memories([mem])
        assert report.inserted == 1

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_block_others(self):
        """单条记忆写入失败 → 记录 failed，其余正常。"""
        emb = _fake_embed("正常")
        good = _make_mem("episodic", "正常记忆", embedding=emb)
        dup_id = str(uuid.uuid4())

        # 先写入一条记忆占据该 ID
        existing = Memory(
            memory_id=dup_id,
            type="episodic",
            content="先占据",
            embedding=emb,
        )
        insert_memory(existing)

        # 构造一条会触发 UNIQUE 约束冲突的记忆（相同 memory_id）
        bad = Memory(
            memory_id=dup_id,  # 重复 ID → 触发 PRIMARY KEY 冲突
            type="episodic",
            content="重复ID记忆",
            embedding=emb,
        )

        report = await write_memories([bad, good])
        assert report.failed >= 1
        # 正常记忆仍被写入
        assert report.inserted >= 1
