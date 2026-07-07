"""
Phase 4 — 记忆合并单元测试。

覆盖：候选筛选纯函数 / 类型资格矩阵 / 贪心配对 / dry-run 零写入 / 端到端。
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
from core.consolidator import (
    ConsolidationReport,
    find_merge_candidates,
)


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """独立临时数据库。"""
    db_file = tmp_path / "test_consolidator.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)
    import core.memory_store as ms
    ms._connection = None
    init_db()
    yield
    close_db()


def _make_mem(
    mem_type: str,
    content: str,
    embedding: np.ndarray | None = None,
    superseded_by: str | None = None,
    unresolved: bool = False,
    decay_weight: float = 1.0,
    access_count: int = 0,
    valence: float = 0.0,
    arousal: float = 0.0,
    created_at: str = "2026-01-01 00:00:00",
    tags: list[str] | None = None,
) -> Memory:
    if embedding is None:
        embedding = np.array([1.0, 0.0], dtype=np.float32)
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=embedding,
        superseded_by=superseded_by,
        unresolved=unresolved,
        decay_weight=decay_weight,
        access_count=access_count,
        valence=valence,
        arousal=arousal,
        created_at=created_at,
        tags=tags or [],
    )


class TestFindMergeCandidates:
    """候选筛选纯函数测试。"""

    def test_episodic_pair_eligible(self):
        """同为 episodic → 入选。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "事件A", emb)
        b = _make_mem("episodic", "事件A v2", emb)  # sim=1.0
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 1

    def test_emotional_pair_eligible(self):
        """同为 emotional → 入选。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("emotional", "情绪A", emb)
        b = _make_mem("emotional", "情绪A v2", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 1

    def test_cross_type_not_eligible(self):
        """episodic × emotional → 不入选。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "事件", emb)
        b = _make_mem("emotional", "情绪", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_semantic_excluded(self):
        """semantic 不参与合并。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("semantic", "语义A", emb)
        b = _make_mem("semantic", "语义A v2", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_procedural_excluded(self):
        """procedural 不参与合并。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("procedural", "规则A", emb)
        b = _make_mem("procedural", "规则A v2", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_superseded_excluded(self):
        """已被覆盖的记忆不参与合并。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "已被覆盖", emb, superseded_by="other-id")
        b = _make_mem("episodic", "正常", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_unresolved_excluded(self):
        """unresolved=True 不参与合并。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "待解决", emb, unresolved=True)
        b = _make_mem("episodic", "正常", emb)
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_no_embedding_excluded(self):
        """无 embedding 不参与合并。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "有向量", emb)
        b = Memory(
            memory_id=str(uuid.uuid4()),
            type="episodic",
            content="无向量",
            embedding=None,
        )
        pairs = find_merge_candidates([a, b])
        assert len(pairs) == 0

    def test_threshold_strictly_greater(self):
        """sim 恰好等于阈值 → 不入选（严格大于）。"""
        # 构造 sim 刚好 ≤ 0.92 的记忆
        # cos(θ) = 0.919, 取 a=[1,0], b=[0.919, sqrt(1-0.919²)]
        import math
        y = math.sqrt(1.0 - 0.919 * 0.919)
        a = _make_mem("episodic", "A",
                      np.array([1.0, 0.0], dtype=np.float32))
        b = _make_mem("episodic", "B",
                      np.array([0.919, float(y)], dtype=np.float32))  # sim ≈ 0.919
        pairs = find_merge_candidates([a, b], threshold=0.92)
        assert len(pairs) == 0  # 0.919 < 0.92，不入选

    def test_threshold_just_above_selected(self):
        """sim=0.93 > 0.92 → 入选。"""
        a = _make_mem("episodic", "A",
                      np.array([1.0, 0.0], dtype=np.float32))
        b = _make_mem("episodic", "B",
                      np.array([0.93, 0.3674], dtype=np.float32))  # sim ≈ 0.93
        pairs = find_merge_candidates([a, b], threshold=0.92)
        assert len(pairs) == 1

    def test_greedy_no_duplicate_usage(self):
        """贪心配对：一条记忆只出现在一个对中。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        a = _make_mem("episodic", "A", emb)
        b = _make_mem("episodic", "B", emb)
        c = _make_mem("episodic", "C", emb)  # 三条都一样，sim=1.0
        pairs = find_merge_candidates([a, b, c])
        # 贪心：A-B 占用了 A 和 B，C 无配对了
        assert len(pairs) == 1

    def test_max_pairs_cap(self):
        """max_pairs 截断生效。"""
        memories = []
        for i in range(10):
            emb = np.array([1.0, float(i) * 0.001], dtype=np.float32)
            a = _make_mem("episodic", f"A{i}", emb)
            b = _make_mem("episodic", f"B{i}", emb)
            memories.extend([a, b])

        pairs = find_merge_candidates(memories, threshold=0.99, max_pairs=3)
        assert len(pairs) <= 3

    def test_similarity_descending_order(self):
        """返回结果按相似度降序。"""
        memories = []
        # 不同相似度的对
        sims = [0.95, 0.99, 0.93]
        for s in sims:
            y = np.sqrt(1.0 - s * s)
            a = _make_mem("episodic", f"A{s}",
                          np.array([1.0, 0.0], dtype=np.float32))
            b = _make_mem("episodic", f"B{s}",
                          np.array([float(s), float(y)], dtype=np.float32))
            memories.extend([a, b])

        pairs = find_merge_candidates(memories, threshold=0.92)
        assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]


class TestConsolidateEndToEnd:
    """端到端合并集成测试（fake LLM + fake embedding）。"""

    @pytest.mark.asyncio
    async def test_consolidate_creates_new_memory(self, monkeypatch, fake_embedding):
        """合并后：新记忆入库、旧两条被 superseded。"""
        import json
        import httpx
        from core.consolidator import consolidate_memories

        # 两条高相似 episodic
        emb = np.array([0.95, 0.312], dtype=np.float32)
        a = _make_mem("episodic", "用户昨天去秋叶原买了机械键盘",
                      embedding=emb, access_count=3, decay_weight=0.8)
        b = _make_mem("episodic", "用户在秋叶原购买了机械键盘",
                      embedding=emb, access_count=2, decay_weight=0.6)

        for m in [a, b]:
            insert_memory(m)

        # fake LLM 返回融合文本
        class FakeConsolidateResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "用户在秋叶原购买了一个机械键盘"}}]}
            @property
            def request(self): return None
            @property
            def text(self): return ""

        async def _fake_post(self, *args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "/v1/embeddings" in url:
                raise RuntimeError("不应调用 embedding")
            return FakeConsolidateResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        report = await consolidate_memories(dry_run=False)
        assert report.merged == 1
        assert report.candidates == 1

        mems = list_active_memories()
        # 新记忆入库
        new_mems = [m for m in mems if m.superseded_by is None]
        assert len(new_mems) == 1
        assert "秋叶原" in new_mems[0].content
        # 旧两条 superseded
        all_mems = list(list_active_memories())  # 重新加载含 superseded
        # 实际上 list_active_memories 只查 superseded_by IS NULL，所以旧两条不会出现
        # 验证一共只有 1 条 active 记忆（新融合的）
        assert len(mems) == 1

        # 回归（insert_memory 曾不落 access_count 列）：合并字段落库后必须保持
        merged = new_mems[0]
        assert merged.access_count == 5                     # 3 + 2（§5.6：两条之和）
        assert merged.decay_weight == pytest.approx(0.8)    # max(0.8, 0.6)
        assert merged.created_at.startswith("2026-01-01")   # 较早一条的时间锚定

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self, fake_embedding):
        """dry_run=True：零写入、零 LLM 调用。"""
        from core.consolidator import consolidate_memories

        emb = np.array([0.95, 0.312], dtype=np.float32)
        a = _make_mem("episodic", "事件A", embedding=emb)
        b = _make_mem("episodic", "事件A v2", embedding=emb)
        for m in [a, b]:
            insert_memory(m)

        mems_before = list_active_memories()
        report = await consolidate_memories(dry_run=True)
        mems_after = list_active_memories()

        assert report.candidates == 1
        assert report.merged == 0  # dry-run 不融合
        assert len(mems_before) == len(mems_after)  # 零变化
        for mb, ma in zip(mems_before, mems_after):
            assert mb.memory_id == ma.memory_id
            assert mb.superseded_by == ma.superseded_by

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, fake_embedding):
        """无候选对时返回空报告。"""
        from core.consolidator import consolidate_memories

        a = _make_mem("episodic", "事件A",
                      embedding=np.array([1.0, 0.0], dtype=np.float32))
        b = _make_mem("episodic", "完全不同的事件B",
                      embedding=np.array([0.0, 1.0], dtype=np.float32))
        for m in [a, b]:
            insert_memory(m)

        report = await consolidate_memories(dry_run=False)
        assert report.candidates == 0
        assert report.merged == 0


class TestCappedAndMetaTimestamp:
    """Bug 修复回归：ConsolidationReport.capped 传出 + last_consolidation_at 用 aware UTC。"""

    def test_find_merge_candidates_capped_counts(self):
        """capped = 超出 max_pairs 的应选对数。"""
        from core.consolidator import find_merge_candidates_capped

        # 3 个正交方向 × 各 2 条相同向量 → 3 个候选对
        mems = []
        for i in range(3):
            emb = np.zeros(3, dtype=np.float32)
            emb[i] = 1.0
            mems.append(_make_mem("episodic", f"事件{i}原文", embedding=emb.copy()))
            mems.append(_make_mem("episodic", f"事件{i}重述", embedding=emb.copy()))

        pairs, capped = find_merge_candidates_capped(mems, max_pairs=2)
        assert len(pairs) == 2
        assert capped == 1

    def test_capped_not_double_counted_for_greedy_blocked_pairs(self):
        """被截断对占用的记忆，其后续贪心冲突对不重复计入 capped。"""
        from core.consolidator import find_merge_candidates_capped

        # 3 条同向记忆 → 3 个原始对 (A,B)(A,C)(B,C)；无上限时贪心只选 1 对
        emb = np.array([1.0, 0.0], dtype=np.float32)
        mems = [
            _make_mem("episodic", f"同一事件表述{i}", embedding=emb.copy())
            for i in range(3)
        ]
        pairs, capped = find_merge_candidates_capped(mems, max_pairs=0)
        assert pairs == []
        assert capped == 1  # 只有 1 对"本会入选"，(A,C)(B,C) 被贪心占用不计

    def test_find_merge_candidates_wrapper_unchanged(self):
        """公开接口 find_merge_candidates 返回值语义不变（只返回列表）。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        mems = [
            _make_mem("episodic", "事件原文", embedding=emb.copy()),
            _make_mem("episodic", "事件重述", embedding=emb.copy()),
        ]
        pairs = find_merge_candidates(mems)
        assert isinstance(pairs, list)
        assert len(pairs) == 1

    @pytest.mark.asyncio
    async def test_report_capped_populated_in_dry_run(self):
        """候选超 CONSOLIDATE_MAX_PAIRS(20) 时 report.capped > 0（dry_run 同样填充）。"""
        from core.consolidator import consolidate_memories

        # 21 个正交方向 × 各 2 条 → 21 个候选对，上限 20
        for i in range(21):
            emb = np.zeros(21, dtype=np.float32)
            emb[i] = 1.0
            insert_memory(_make_mem("episodic", f"事件{i}原文", embedding=emb.copy()))
            insert_memory(_make_mem("episodic", f"事件{i}重述", embedding=emb.copy()))

        report = await consolidate_memories(dry_run=True)
        assert report.candidates == 20
        assert report.capped == 1

    @pytest.mark.asyncio
    async def test_last_consolidation_at_is_aware_utc(self, monkeypatch, fake_embedding):
        """meta.last_consolidation_at 为 aware UTC ISO 时间戳（禁用裸 datetime.now()）。"""
        from datetime import datetime

        from core.consolidator import consolidate_memories
        from core.memory_store import get_meta

        emb = np.array([0.95, 0.312], dtype=np.float32)
        insert_memory(_make_mem("episodic", "事件A", embedding=emb.copy()))
        insert_memory(_make_mem("episodic", "事件A v2", embedding=emb.copy()))

        async def _fake_merge(content_a, content_b, mem_type):
            return "融合后的事件A"

        monkeypatch.setattr("core.consolidator._llm_merge_pair", _fake_merge)

        report = await consolidate_memories(dry_run=False)
        assert report.merged == 1

        val = get_meta("last_consolidation_at")
        assert val is not None
        parsed = datetime.fromisoformat(val)
        assert parsed.tzinfo is not None                       # aware
        assert parsed.utcoffset().total_seconds() == 0         # UTC


# fake embedding fixture（与 integration 测试共享模式）
@pytest.fixture
def fake_embedding(monkeypatch):
    """Fake embedding 返回固定向量。"""
    import core.embedding as emb_mod
    import core.dehydrator as dehydrator_mod

    async def _fake_embed(text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("空文本")
        seed = sum(ord(c) for c in text)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1536).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    monkeypatch.setattr(emb_mod, "embed_text", _fake_embed)
    monkeypatch.setattr(dehydrator_mod, "embed_text", _fake_embed)
    # also patch consolidator's embed_text
    monkeypatch.setattr("core.consolidator.embed_text", _fake_embed)
