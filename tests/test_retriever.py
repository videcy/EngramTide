"""
测试检索器：排序、top_k、空数据库、缺失 embedding。
"""

import uuid

import numpy as np
import pytest

from core.memory_store import Memory, close_db, init_db, insert_memory, list_active_memories
from core.retriever import retrieve_memories


def _make_mem(
    content: str,
    embedding: np.ndarray | None,
    mem_type: str = "semantic",
    decay_weight: float = 1.0,
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=embedding,
        decay_weight=decay_weight,
    )


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """独立临时数据库。"""
    db_file = tmp_path / "test_retriever.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)

    import core.memory_store as ms

    ms._connection = None
    init_db()
    yield
    close_db()


def test_retrieve_top_k_ordering():
    """最相似的记忆应当排在最前面。"""
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # 插入三条人工记忆
    close_match = _make_mem("接近", np.array([0.9, 0.1, 0.0], dtype=np.float32))
    mid_match = _make_mem("中等", np.array([0.5, 0.5, 0.0], dtype=np.float32))
    far_match = _make_mem("远离", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    for m in [close_match, mid_match, far_match]:
        insert_memory(m)

    results = retrieve_memories(query_vec, top_k=3)
    assert len(results) == 3
    # close 应该第一，far 应该最后
    assert results[0][0].content == "接近"
    assert results[2][0].content == "远离"
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_top_k_limit():
    """top_k 参数生效。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)

    for i in range(10):
        insert_memory(
            _make_mem(f"记忆#{i}", np.array([1.0, float(i) * 0.1], dtype=np.float32))
        )

    results = retrieve_memories(query_vec, top_k=3)
    assert len(results) == 3


def test_empty_database_returns_empty():
    """空数据库返回空列表。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    results = retrieve_memories(query_vec)
    assert results == []


def test_missing_embedding_skipped():
    """缺失 embedding 的记忆不会导致崩溃。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)

    insert_memory(_make_mem("无 embedding", None))
    insert_memory(
        _make_mem("有 embedding", np.array([1.0, 0.0], dtype=np.float32))
    )

    results = retrieve_memories(query_vec)
    assert len(results) == 1
    assert results[0][0].content == "有 embedding"


def test_superseded_memory_excluded_from_retrieval():
    """被取代的记忆不在检索结果中出现。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    active_id = str(uuid.uuid4())

    insert_memory(
        Memory(
            memory_id=active_id,
            type="semantic",
            content="活跃",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
    )
    insert_memory(
        Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content="已取代",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            superseded_by=active_id,
        )
    )

    results = retrieve_memories(query_vec)
    assert len(results) == 1
    assert results[0][0].content == "活跃"


def test_zero_decay_excluded():
    """decay_weight <= 0 的记忆被跳过。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    emb = np.array([1.0, 0.0], dtype=np.float32)

    insert_memory(
        Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content="衰减为0",
            embedding=emb,
            decay_weight=0.0,
        )
    )
    insert_memory(
        Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content="正常",
            embedding=emb,
            decay_weight=1.0,
        )
    )

    results = retrieve_memories(query_vec)
    assert len(results) == 1
    assert results[0][0].content == "正常"


def test_retrieve_does_not_mark_accessed_before_context():
    """retriever 只排序，不提前更新访问统计。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    insert_memory(_make_mem("相关", np.array([1.0, 0.0], dtype=np.float32)))

    results = retrieve_memories(query_vec, top_k=1)
    assert len(results) == 1

    memories = list_active_memories()
    assert memories[0].access_count == 0


def test_decay_below_min_threshold_excluded():
    """decay_weight < RETRIEVAL_MIN_DECAY 的记忆被排除。"""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    emb = np.array([1.0, 0.0], dtype=np.float32)

    # 衰减刚好在阈值之下的记忆
    insert_memory(
        Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content="衰减过低 (0.09)",
            embedding=emb,
            decay_weight=0.09,
        )
    )
    # 衰减刚好在阈值之上的记忆
    insert_memory(
        Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content="刚好达标 (0.1)",
            embedding=emb,
            decay_weight=0.1,
        )
    )

    results = retrieve_memories(query_vec)
    assert len(results) == 1
    assert results[0][0].content == "刚好达标 (0.1)"


# ════════════════════════════════════════════════════════════
# Phase 4：双通道检索测试
# ════════════════════════════════════════════════════════════


class TestDualChannelRetrieval:
    """双通道检索：关键词拉回、通道关闭等价性、沉底不绕过。"""

    def test_keyword_channel_pulls_exact_match(self):
        """字面命中优先于语义接近但字面无关的记忆。"""
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_text = "rapidfuzz 怎么配置来着"

        # 目标：字面含 rapidfuzz（vec sim≈0.5、kw 高 → 双通道 0.7·0.5+0.3·~0.7 ≈ 0.56）
        target = _make_mem(
            "项目用了 rapidfuzz 做模糊匹配",
            np.array([0.5, 0.866, 0.0], dtype=np.float32),
        )
        # 干扰：语义接近但字面无关（vec sim≈0.52 → 纯向量 0.52）
        distractor = _make_mem(
            "项目用了模糊字符串匹配库",
            np.array([0.52, 0.854, 0.0], dtype=np.float32),
        )

        for m in [distractor, target]:
            insert_memory(m)

        results = retrieve_memories(query_vec, top_k=2, query_text=query_text)
        # 双通道下目标应排第一
        assert results[0][0].content == target.content

    def test_channel_disabled_returns_vector_only(self):
        """KEYWORD_CHANNEL_ENABLED=False 时退化为纯向量。"""
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_text = "rapidfuzz"

        target = _make_mem(
            "项目用了 rapidfuzz",
            np.array([0.5, 0.866, 0.0], dtype=np.float32),  # sim ≈ 0.5
        )
        distractor = _make_mem(
            "项目用了模糊字符串匹配",
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # sim ≈ 1.0
        )
        for m in [target, distractor]:
            insert_memory(m)

        # 关闭关键词通道
        import config
        old_val = config.KEYWORD_CHANNEL_ENABLED
        config.KEYWORD_CHANNEL_ENABLED = False
        try:
            results = retrieve_memories(query_vec, top_k=2, query_text=query_text)
            # 纯向量：sim 高的排第一
            assert results[0][0].content == distractor.content
        finally:
            config.KEYWORD_CHANNEL_ENABLED = old_val

    def test_query_text_none_returns_phase3_equivalent(self):
        """query_text=None 时与 Phase 3 打分逐位一致（无权重缩放）。"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        emb = np.array([0.9, 0.1], dtype=np.float32)
        insert_memory(_make_mem("测试记忆", emb, decay_weight=0.8))
        insert_memory(_make_mem("另一记忆", np.array([0.5, 0.5], dtype=np.float32), decay_weight=1.0))

        results_none = retrieve_memories(query_vec, top_k=5, query_text=None)
        results_with = retrieve_memories(query_vec, top_k=5, query_text="测试")

        # query_text=None 时分数应与纯向量一致
        # 手动计算纯向量分验证
        from utils.similarity import cosine_similarity
        for (mem, score), _ in zip(results_none, range(2)):
            expected = cosine_similarity(query_vec, mem.embedding) * mem.decay_weight
            assert abs(score - expected) < 1e-9

    def test_sunk_memory_not_bypassed_by_keyword(self):
        """沉底记忆（decay < RETRIEVAL_MIN_DECAY）即使字面全匹配也不出现。"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        query_text = "sunk记忆"

        insert_memory(
            Memory(
                memory_id=str(uuid.uuid4()),
                type="episodic",
                content="sunk记忆内容完全匹配",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                decay_weight=0.05,  # < 0.1
            )
        )

        results = retrieve_memories(query_vec, top_k=5, query_text=query_text)
        assert len(results) == 0

    def test_keyword_channel_scores_math(self):
        """验证双通道打分公式：(0.7·sim + 0.3·kw)·decay_weight。"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        query_text = "测试关键词"

        # 构造确定性：embedding 完全相同 → sim=1.0
        emb = np.array([1.0, 0.0], dtype=np.float32)
        mem = _make_mem("测试关键词匹配", emb, decay_weight=0.8)
        insert_memory(mem)

        results = retrieve_memories(query_vec, top_k=1, query_text=query_text)
        score = results[0][1]

        # kw = partial_ratio("测试关键词", "测试关键词匹配") / 100
        from rapidfuzz import fuzz
        kw = fuzz.partial_ratio(query_text, mem.content) / 100.0
        expected = (0.7 * 1.0 + 0.3 * kw) * 0.8
        assert abs(score - expected) < 1e-9

    def test_short_query_does_not_crash(self):
        """极短 query（1-2 字）不崩溃。"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(_make_mem("长内容记忆", np.array([0.9, 0.1], dtype=np.float32)))

        results = retrieve_memories(query_vec, top_k=5, query_text="测")
        assert isinstance(results, list)

    def test_superseded_excluded_even_with_keyword_hit(self):
        """被取代的记忆字面命中也不出现。"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        active_id = str(uuid.uuid4())

        insert_memory(
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content="被取代的 rapidfuzz 记忆",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                superseded_by=active_id,
            )
        )

        results = retrieve_memories(query_vec, top_k=5, query_text="rapidfuzz")
        assert len(results) == 0


# ════════════════════════════════════════════════════════════
# Phase 4 次要项修复回归：检索分数分解（计划 §9.2）
# ════════════════════════════════════════════════════════════


class TestRetrievalDetail:
    """retrieve_memories_detailed 的分解语义与包装等价性。"""

    def test_detailed_decomposition_math(self):
        """通道开启时：kw 有值，score == (0.7·sim + 0.3·kw)·w，sim 为原始余弦。"""
        from rapidfuzz import fuzz

        from core.retriever import retrieve_memories_detailed
        from utils.similarity import cosine_similarity

        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        query_text = "测试关键词"
        emb = np.array([0.9, 0.436], dtype=np.float32)
        mem = _make_mem("测试关键词匹配", emb, decay_weight=0.8)
        insert_memory(mem)

        details = retrieve_memories_detailed(query_vec, top_k=1, query_text=query_text)
        assert len(details) == 1
        d = details[0]

        expected_sim = cosine_similarity(query_vec, emb)
        expected_kw = fuzz.partial_ratio(query_text, mem.content) / 100.0
        assert abs(d.sim - expected_sim) < 1e-9
        assert d.kw is not None
        assert abs(d.kw - expected_kw) < 1e-9
        assert abs(d.score - (0.7 * d.sim + 0.3 * d.kw) * 0.8) < 1e-9

    def test_detailed_kw_none_when_channel_inactive(self):
        """query_text=None → kw 为 None，score 为纯向量分。"""
        from core.retriever import retrieve_memories_detailed
        from utils.similarity import cosine_similarity

        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        emb = np.array([0.9, 0.436], dtype=np.float32)
        insert_memory(_make_mem("纯向量记忆", emb, decay_weight=0.5))

        details = retrieve_memories_detailed(query_vec, top_k=1, query_text=None)
        d = details[0]
        assert d.kw is None
        assert abs(d.score - cosine_similarity(query_vec, emb) * 0.5) < 1e-9

    def test_wrapper_equals_detailed(self):
        """retrieve_memories 与 detailed 版提取 (memory, score) 后逐位一致。"""
        from core.retriever import retrieve_memories_detailed

        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        for i, x in enumerate([0.9, 0.7, 0.5]):
            emb = np.array([x, np.sqrt(1 - x * x)], dtype=np.float32)
            insert_memory(_make_mem(f"记忆{i}", emb))

        wrapped = retrieve_memories(query_vec, top_k=3, query_text="记忆")
        detailed = retrieve_memories_detailed(query_vec, top_k=3, query_text="记忆")

        assert len(wrapped) == len(detailed) == 3
        for (mem_w, score_w), d in zip(wrapped, detailed):
            assert mem_w.memory_id == d.memory.memory_id
            assert abs(score_w - d.score) < 1e-12
