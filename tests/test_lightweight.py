"""
轻量化改造（P0–P4）测试。

覆盖：
- P0 埋点开关、批量落库、保留期清理
- P1 向量缓存、失效策略、**共用加载的权重同步对照测试**
- P2 FTS5 关键词通道与 bm25 归一化
- P3 embedding 归一化 / MRL 截断 / 维度守卫
- P4 episodic 去重、冷归档、supersede 清理、/stats
"""

import uuid
from datetime import timedelta

import numpy as np
import pytest

import config
from core.decay import context_aware_update
from core.memory_store import (
    Memory,
    close_db,
    collect_archive_candidates,
    archive_memories,
    data_version,
    fts_available,
    fts_search,
    init_db,
    insert_memories,
    insert_memory,
    list_active_memories,
    list_archived_memories,
    log_access_events_batch,
    log_decay_events_batch,
    mark_superseded,
    memory_stats,
    prune_event_logs,
    purge_superseded,
    query_access_events,
    restore_archived,
)
from core.retriever import retrieve_memories_detailed
from core.vector_index import VectorCache, reset_cache, similarity_map
from utils.similarity import cosine_similarity
from utils.time_utils import utc_now


def _vec(*values, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(values):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


def _mem(
    mem_type: str = "episodic",
    content: str = "内容",
    embedding: np.ndarray | None = None,
    decay_weight: float = 1.0,
    access_count: int = 0,
    created_at: str = "",
    last_accessed: str = "",
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=embedding,
        decay_weight=decay_weight,
        access_count=access_count,
        created_at=created_at,
        last_accessed=last_accessed,
    )


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """独立临时库；每个用例前后都丢弃进程内向量缓存。"""
    import core.memory_store as ms

    monkeypatch.setattr(ms, "DB_PATH", tmp_path / "test_lightweight.db")
    ms._connection = None
    reset_cache()
    init_db()
    yield
    close_db()
    reset_cache()


# ════════════════════════════════════════════════════════════
# P0 埋点
# ════════════════════════════════════════════════════════════


class TestEventLogging:
    def test_decay_log_disabled_by_default(self):
        """生产默认不写 decay_events —— 这是 P0 的全部意义。"""
        assert config.DECAY_LOG_ENABLED is False
        assert config.ACCESS_LOG_ENABLED is False

    def test_batch_write_is_single_transaction(self):
        rows = [(str(uuid.uuid4()), utc_now().isoformat(), 1.0, 0, 0.9, 0) for _ in range(50)]
        log_decay_events_batch(rows)

        from core.memory_store import _get_conn

        assert _get_conn().execute("SELECT COUNT(*) FROM decay_events").fetchone()[0] == 50

    def test_batch_write_empty_is_noop(self):
        log_decay_events_batch([])
        log_access_events_batch([])

    def test_access_events_batch_records_source(self):
        mid = str(uuid.uuid4())
        log_access_events_batch([(mid, "retrieval"), (mid, "surface")])
        events = query_access_events(mid)
        assert [e["source"] for e in events] == ["retrieval", "surface"]

    def test_prune_removes_only_expired_rows(self):
        old_ts = (utc_now() - timedelta(days=90)).isoformat()
        fresh_ts = utc_now().isoformat()
        log_decay_events_batch(
            [("old", old_ts, 1.0, 0, 0.9, 0), ("new", fresh_ts, 1.0, 0, 0.9, 0)]
        )

        assert prune_event_logs(retention_days=30) == 1

        from core.memory_store import _get_conn

        remaining = _get_conn().execute(
            "SELECT memory_id FROM decay_events"
        ).fetchall()
        assert [r["memory_id"] for r in remaining] == ["new"]

    def test_prune_disabled_when_retention_non_positive(self):
        log_decay_events_batch([("x", (utc_now() - timedelta(days=999)).isoformat(),
                                1.0, 0, 0.9, 0)])
        assert prune_event_logs(retention_days=0) == 0


# ════════════════════════════════════════════════════════════
# P1 向量缓存
# ════════════════════════════════════════════════════════════


class TestVectorCache:
    def test_matches_pairwise_cosine(self):
        """矩阵乘的结果必须与逐条 cosine_similarity 一致（float32 精度内）。"""
        rng = np.random.RandomState(7)
        mems = [
            _mem(embedding=rng.randn(64).astype(np.float32)) for _ in range(30)
        ]
        query = rng.randn(64).astype(np.float32)

        sims = similarity_map(query, mems)
        for m in mems:
            assert sims[m.memory_id] == pytest.approx(
                cosine_similarity(query, m.embedding), abs=1e-5
            )

    def test_unnormalised_legacy_vectors_still_correct(self):
        """旧库里未归一化的向量也要给出正确余弦（重建时按行归一化）。"""
        m = _mem(embedding=np.array([3.0, 4.0] + [0.0] * 6, dtype=np.float32))
        sims = similarity_map(_vec(1.0), [m])
        assert sims[m.memory_id] == pytest.approx(0.6, abs=1e-6)

    def test_cache_reused_when_membership_unchanged(self):
        mems = [_mem(embedding=_vec(1.0)) for _ in range(3)]
        cache = VectorCache()
        cache.rebuild(mems)
        assert cache.is_fresh(mems)

    def test_insert_invalidates_cache(self):
        """集合成员变化 → 缓存失效。"""
        mems = [_mem(embedding=_vec(1.0))]
        cache = VectorCache()
        cache.rebuild(mems)
        insert_memory(_mem(embedding=_vec(1.0)))
        assert not cache.is_fresh(mems)

    def test_decay_weight_change_does_not_invalidate(self):
        """★ 权重变化不碰矩阵——否则缓存每轮重建，P1 的收益归零。"""
        from core.memory_store import update_decay_weights

        m = _mem(embedding=_vec(1.0))
        insert_memory(m)
        mems = list_active_memories()
        cache = VectorCache()
        cache.rebuild(mems)

        update_decay_weights([(0.5, m.memory_id)])
        assert cache.is_fresh(mems)

    def test_mixed_dimensions_are_dropped_not_crashed(self):
        good = [_mem(embedding=_vec(1.0, dim=8)) for _ in range(3)]
        odd = _mem(embedding=np.ones(4, dtype=np.float32))
        sims = similarity_map(_vec(1.0, dim=8), good + [odd])
        assert odd.memory_id not in sims
        assert len(sims) == 3

    def test_query_dimension_mismatch_returns_zeros(self):
        m = _mem(embedding=_vec(1.0, dim=8))
        sims = similarity_map(np.ones(16, dtype=np.float32), [m])
        assert sims[m.memory_id] == 0.0

    def test_memories_without_embedding_are_absent(self):
        m = _mem(embedding=None)
        assert similarity_map(_vec(1.0), [m]) == {}


class TestSharedLoadEquivalence:
    """
    ★ P1 唯一的正确性风险点，对照测试。

    改造前：激活把新权重落库 → 检索重新读库 → 读到的是**新**权重。
    改造后：两者共用同一个列表，如果只落库不同步内存对象，检索会读到**旧**权重，
    Top-K 结果随之改变。这条用例锁死「同一输入下 Top-K 的 memory_id 序列不变」。
    """

    @staticmethod
    def _bank():
        """构造一批与 query 相似度分散、权重也分散的记忆。"""
        query = _vec(1.0)
        mems = []
        for i in range(8):
            angle = i * 0.12
            emb = _vec(np.cos(angle), np.sin(angle))
            mems.append(
                _mem(
                    content=f"记忆-{i}",
                    embedding=emb,
                    decay_weight=0.3 + 0.05 * i,
                )
            )
        insert_memories(mems)
        return query

    def test_topk_identical_to_reload_from_db(self):
        query = self._bank()

        # 改造后：一次加载 + 共用相似度
        turn_memories = list_active_memories()
        sims = similarity_map(query, turn_memories)
        context_aware_update(query, memories=turn_memories, sims=sims)
        new_ids = [
            d.memory.memory_id
            for d in retrieve_memories_detailed(
                query, memories=turn_memories, top_k=5, query_text=None, sims=sims
            )
        ]

        # 改造前的等价物：激活已落库，检索重新读库
        reloaded = list_active_memories()
        legacy_ids = [
            d.memory.memory_id
            for d in retrieve_memories_detailed(
                query, memories=reloaded, top_k=5, query_text=None
            )
        ]

        assert new_ids == legacy_ids

    def test_weights_are_written_back_into_the_shared_list(self):
        query = self._bank()
        turn_memories = list_active_memories()
        before = {m.memory_id: m.decay_weight for m in turn_memories}

        context_aware_update(query, memories=turn_memories)

        persisted = {m.memory_id: m.decay_weight for m in list_active_memories()}
        in_memory = {m.memory_id: m.decay_weight for m in turn_memories}
        assert in_memory == persisted
        assert in_memory != before      # 确认这一轮确实发生了激活（非空转）


# ════════════════════════════════════════════════════════════
# P2 FTS5
# ════════════════════════════════════════════════════════════


class TestFtsKeywordChannel:
    """FTS5 可用性只有在 init_db() 之后才知道，所以在用例里跳过而不是收集期跳过。"""

    @pytest.fixture(autouse=True)
    def require_fts(self):
        if not fts_available():
            pytest.skip("本机 SQLite 未编译 FTS5")

    def test_cjk_content_is_indexed(self):
        """trigram 分词器对中文有效——unicode61 会把整句切成一个 token。"""
        m = _mem(content="我下周二要去字节跳动面试实习", embedding=_vec(1.0))
        insert_memory(m)
        assert m.memory_id in fts_search("字节跳动", 10)

    def test_scores_are_normalised_into_unit_range(self):
        """
        bm25 是无上界的负数，而双通道公式假设 kw ∈ [0, 1]。
        不归一化就会静默改变 RETRIEVAL_KEYWORD_WEIGHT 的实际含义。
        """
        for i in range(5):
            insert_memory(_mem(content=f"Tableau 仪表盘 作业 {i}", embedding=_vec(1.0)))
        scores = fts_search("Tableau 仪表盘", 10).values()
        assert scores
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_misses_are_not_scored(self):
        hit = _mem(content="秋叶原买了青轴机械键盘", embedding=_vec(1.0))
        miss = _mem(content="今天中午吃了麻辣烫", embedding=_vec(1.0))
        insert_memories([hit, miss])
        result = fts_search("机械键盘", 10)
        assert hit.memory_id in result
        assert miss.memory_id not in result

    def test_deleted_memory_leaves_the_index(self):
        from core.memory_store import delete_memories

        m = _mem(content="需要被删除的独特内容 zzz", embedding=_vec(1.0))
        insert_memory(m)
        assert fts_search("独特内容", 10)
        delete_memories([m.memory_id])
        assert fts_search("独特内容", 10) == {}

    def test_updated_content_is_reindexed(self):
        """UPDATE 触发器：改了 content，旧词不该再命中、新词该命中。"""
        from core.memory_store import _get_conn

        m = _mem(content="原始的独特字样", embedding=_vec(1.0))
        insert_memory(m)
        conn = _get_conn()
        with conn:
            conn.execute(
                "UPDATE memories SET content = ? WHERE memory_id = ?",
                ("替换后的独特字样", m.memory_id),
            )
        assert fts_search("原始的", 10) == {}
        assert m.memory_id in fts_search("替换后", 10)

    def test_query_syntax_characters_do_not_crash(self):
        """用户输入里的 FTS5 语法字符必须被当成字面量。"""
        insert_memory(_mem(content="正常内容", embedding=_vec(1.0)))
        for q in ('"', "NEAR(a b)", "foo* AND", "a:b", "()"):
            fts_search(q, 10)

    def test_two_character_chinese_word_falls_back(self):
        """
        ★ trigram 要求检索项 ≥3 字符，而中文常用词多是 2 个字。
        这类查询 FTS5 处理不了 → 返回 None，让调用方降级到模糊匹配，
        而不是当成「没有关键词命中」把通道静默关掉。
        """
        insert_memory(_mem(content="东京的地铁末班车", embedding=_vec(1.0)))
        assert fts_search("东京", 10) is None

    def test_long_chinese_sentence_matches_by_trigram(self):
        """整句不能当一个短语查（那只剩精确子串匹配），要拆成滑动 3-gram。"""
        m = _mem(content="我上次在东京吃的那家拉面店", embedding=_vec(1.0))
        insert_memory(m)
        assert m.memory_id in fts_search("东京拉面店在哪", 10)

    def test_unmatchable_query_degrades_to_fuzzy(self):
        """FTS 处理不了时，关键词分应由 rapidfuzz 给出，而不是一律 0。"""
        m = _mem(content="东京的地铁末班车", embedding=_vec(1.0))
        insert_memory(m)
        detail = retrieve_memories_detailed(
            _vec(1.0), memories=list_active_memories(), top_k=1, query_text="东京"
        )[0]
        assert detail.kw is not None and detail.kw > 0

    def test_keyword_channel_rescues_low_similarity_exact_match(self):
        """
        关键词通道存在的意义：专有名词命中但语义相似度低时的兜底。
        这条用例保证换成 FTS5 之后这个能力没丢。
        """
        far = _vec(0.0, 1.0)
        near = _vec(1.0)
        名词命中 = _mem(content="项目用了 Milvus 做向量检索", embedding=far)
        语义接近 = _mem(content="完全无关的另一条记忆", embedding=near)
        insert_memories([名词命中, 语义接近])

        details = retrieve_memories_detailed(
            near, memories=list_active_memories(), top_k=2, query_text="Milvus"
        )
        by_id = {d.memory.memory_id: d for d in details}
        assert by_id[名词命中.memory_id].kw > 0
        assert by_id[语义接近.memory_id].kw == 0


class TestRetrieverDegradation:
    def test_keyword_channel_off_is_pure_vector(self, monkeypatch):
        monkeypatch.setattr("core.retriever.KEYWORD_CHANNEL_ENABLED", False)
        m = _mem(content="任意内容", embedding=_vec(1.0), decay_weight=0.5)
        insert_memory(m)
        detail = retrieve_memories_detailed(
            _vec(1.0), memories=list_active_memories(), top_k=1, query_text="任意"
        )[0]
        assert detail.kw is None
        assert detail.score == pytest.approx(detail.sim * 0.5)


# ════════════════════════════════════════════════════════════
# P3 embedding
# ════════════════════════════════════════════════════════════


class TestEmbeddingShape:
    def test_response_is_truncated_and_normalised(self):
        from core.embedding import _parse_embedding_response, reset_dim_state

        reset_dim_state()
        raw = list(np.arange(1, config.EMBEDDING_DIM * 3 + 1, dtype=float))
        vec = _parse_embedding_response({"data": [{"embedding": raw}]})

        assert vec.shape[0] == config.EMBEDDING_DIM
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)
        reset_dim_state()

    def test_shorter_vector_is_left_alone(self):
        from core.embedding import _truncate_to_dim

        short = np.array([3.0, 4.0], dtype=np.float32)
        out = _truncate_to_dim(short)
        assert out.shape[0] == 2
        assert float(np.linalg.norm(out)) == pytest.approx(1.0)

    def test_payload_carries_dimensions(self):
        from core.embedding import _build_payload

        assert _build_payload("x", True)["dimensions"] == config.EMBEDDING_DIM
        assert "dimensions" not in _build_payload("x", False)

    def test_dim_guard_reports_mismatch(self, monkeypatch):
        from core.memory_store import set_meta

        set_meta("embedding_dim", str(config.EMBEDDING_DIM + 128))
        monkeypatch.setattr(config, "DB_PATH", __import__("core.memory_store",
                                                          fromlist=["DB_PATH"]).DB_PATH)
        error = config.check_embedding_dim()
        assert error is not None
        assert "rebuild_embeddings" in error

    def test_dim_guard_silent_when_consistent(self, monkeypatch):
        from core.memory_store import set_meta

        set_meta("embedding_dim", str(config.EMBEDDING_DIM))
        monkeypatch.setattr(config, "DB_PATH", __import__("core.memory_store",
                                                          fromlist=["DB_PATH"]).DB_PATH)
        assert config.check_embedding_dim() is None


# ════════════════════════════════════════════════════════════
# P4 增长治理
# ════════════════════════════════════════════════════════════


class TestBatchInsert:
    def test_default_timestamps_are_filled(self):
        """批量路径不能把 created_at 写成 NULL —— DEFAULT 只在省略列时生效。"""
        insert_memories([_mem(embedding=_vec(1.0)) for _ in range(3)])
        for m in list_active_memories():
            assert m.created_at
            assert m.last_accessed

    def test_explicit_timestamps_are_preserved(self):
        ts = "2026-01-02 03:04:05"
        insert_memories([_mem(embedding=_vec(1.0), created_at=ts, last_accessed=ts)])
        assert list_active_memories()[0].created_at == ts

    def test_duplicate_id_does_not_lose_the_others(self):
        dup = str(uuid.uuid4())
        a = _mem(content="A", embedding=_vec(1.0))
        a.memory_id = dup
        insert_memory(a)

        b = _mem(content="B", embedding=_vec(1.0))
        b.memory_id = dup            # 主键冲突
        good = _mem(content="C", embedding=_vec(1.0))
        insert_memories([b, good])

        contents = {m.content for m in list_active_memories()}
        assert contents == {"A", "C"}

    def test_data_version_advances_once_per_batch(self):
        before = data_version()
        insert_memories([_mem(embedding=_vec(1.0)) for _ in range(5)])
        assert data_version() == before + 1


class TestArchiving:
    @staticmethod
    def _stale(**kwargs):
        long_ago = (utc_now() - timedelta(days=200)).isoformat()
        params = dict(
            mem_type="episodic",
            embedding=_vec(1.0),
            decay_weight=config.DECAY_FLOOR,
            access_count=0,
            created_at=long_ago,
            last_accessed=long_ago,
        )
        params.update(kwargs)
        return _mem(**params)

    def test_sunken_untouched_episodic_is_a_candidate(self):
        m = self._stale()
        insert_memory(m)
        assert collect_archive_candidates() == [m.memory_id]

    def test_used_memory_is_never_archived(self):
        """★ access_count > 0 永不归档——被真正用过的东西留在热表，这是刻意的偏置。"""
        insert_memory(self._stale(access_count=1))
        assert collect_archive_candidates() == []

    def test_recently_accessed_is_not_archived(self):
        insert_memory(self._stale(last_accessed=utc_now().isoformat()))
        assert collect_archive_candidates() == []

    def test_non_sunken_is_not_archived(self):
        insert_memory(self._stale(decay_weight=0.4))
        assert collect_archive_candidates() == []

    def test_only_episodic_is_archived(self):
        for t in ("semantic", "emotional", "procedural"):
            insert_memory(self._stale(mem_type=t))
        assert collect_archive_candidates() == []

    def test_archive_moves_row_out_of_hot_table(self):
        m = self._stale()
        insert_memory(m)
        assert archive_memories([m.memory_id]) == 1
        assert list_active_memories() == []
        archived = list_archived_memories()
        assert [a.memory_id for a in archived] == [m.memory_id]
        assert archived[0].content == m.content

    def test_archive_preserves_embedding(self):
        """归档不是删除：向量还在，restore 后还能继续参与检索。"""
        m = self._stale()
        insert_memory(m)
        archive_memories([m.memory_id])
        assert list_archived_memories()[0].embedding is not None

    def test_restore_brings_it_back(self):
        m = self._stale()
        insert_memory(m)
        archive_memories([m.memory_id])
        assert restore_archived([m.memory_id]) == 1
        assert [x.memory_id for x in list_active_memories()] == [m.memory_id]
        assert list_archived_memories() == []

    def test_archive_invalidates_vector_cache(self):
        m = self._stale()
        insert_memory(m)
        mems = list_active_memories()
        cache = VectorCache()
        cache.rebuild(mems)
        archive_memories([m.memory_id])
        assert not cache.is_fresh(mems)

    def test_search_archive_finds_content(self):
        from core.maintenance import search_archive

        m = self._stale(content="被归档的独特事件")
        insert_memory(m)
        archive_memories([m.memory_id])
        assert [h.memory_id for h in search_archive("独特事件")] == [m.memory_id]


class TestSupersededPurge:
    def test_expired_superseded_is_hard_deleted(self):
        old = _mem(mem_type="semantic", embedding=_vec(1.0))
        new = _mem(mem_type="semantic", embedding=_vec(1.0))
        insert_memories([old, new])
        mark_superseded([old.memory_id], new.memory_id)

        assert purge_superseded(retention_days=0,
                                now=utc_now() + timedelta(seconds=1)) == 1
        assert [m.memory_id for m in list_active_memories()] == [new.memory_id]

    def test_fresh_superseded_is_kept(self):
        old = _mem(mem_type="semantic", embedding=_vec(1.0))
        new = _mem(mem_type="semantic", embedding=_vec(1.0))
        insert_memories([old, new])
        mark_superseded([old.memory_id], new.memory_id)
        assert purge_superseded(retention_days=30) == 0


class TestMaintenanceScheduling:
    def test_runs_only_every_n_sessions(self, monkeypatch):
        from core.maintenance import run_maintenance

        monkeypatch.setattr(config, "ARCHIVE_RUN_EVERY_SESSIONS", 3)
        assert [run_maintenance().ran for _ in range(3)] == [False, False, True]

    def test_force_runs_immediately(self, monkeypatch):
        from core.maintenance import run_maintenance

        monkeypatch.setattr(config, "ARCHIVE_RUN_EVERY_SESSIONS", 100)
        assert run_maintenance(force=True).ran is True

    def test_archive_disabled_still_prunes_events(self, monkeypatch):
        from core.maintenance import run_maintenance

        monkeypatch.setattr(config, "ARCHIVE_ENABLED", False)
        stale = TestArchiving._stale()
        insert_memory(stale)
        log_decay_events_batch(
            [("x", (utc_now() - timedelta(days=999)).isoformat(), 1.0, 0, 0.9, 0)]
        )

        report = run_maintenance(force=True)
        assert report.pruned_events == 1
        assert report.archived == 0
        assert len(list_active_memories()) == 1


class TestStats:
    def test_counts_and_net_growth(self):
        insert_memories([
            _mem("semantic", "a", _vec(1.0)),
            _mem("episodic", "b", _vec(1.0)),
            _mem("episodic", "c", _vec(1.0)),
        ])
        st = memory_stats()
        assert st["active_total"] == 3
        assert st["active_by_type"]["episodic"] == 2
        assert st["new_last_30d"] == 3
        assert st["net_growth_last_30d"] == 3

    def test_archived_rows_leave_the_active_count(self):
        m = TestArchiving._stale()
        insert_memory(m)
        archive_memories([m.memory_id])
        st = memory_stats()
        assert st["active_total"] == 0
        assert st["archived"] == 1
        # 归档的是 200 天前建的旧记忆：本月没有新增、却归档了 1 条 → 净增长 -1
        assert st["net_growth_last_30d"] == -1
