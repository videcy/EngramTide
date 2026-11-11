"""
测试写入管线：四类写入策略（semantic 覆盖 / emotional 强化 / procedural 去重 / episodic 直写）。
"""

import uuid

import numpy as np
import pytest

import config
import core.memory_writer as mw
from core.conflict import Relation
from core.memory_store import (
    Memory,
    close_db,
    get_memory,
    init_db,
    insert_memory,
    links_for,
    list_active_memories,
    list_archived_memories,
    list_pending_links,
)
from config import EPISODIC_REINFORCE_BOOST
from core.memory_writer import write_memories, WriteReport


def _make_mem(
    mem_type: str,
    content: str,
    embedding: np.ndarray | None = None,
    valence: float = 0.0,
    arousal: float = 0.0,
    tags: list[str] | None = None,
    decay_weight: float = 1.0,
    **extra,
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
        **extra,
    )


# 确定性的 fake embedding：内容不同 → 向量不同（用 hash 种子）
def _fake_embed(text: str) -> np.ndarray:
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    vec = rng.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture(autouse=True)
def no_llm_judge(monkeypatch):
    """默认只走规则判定，保证不发真实 LLM 请求。"""
    monkeypatch.setattr(config, "CONFLICT_LLM_JUDGE_ENABLED", False)


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
    async def test_episodic_near_duplicate_reinforces(self):
        """P4-①：近重复 episodic 强化已有记忆，不新增条目。"""
        emb = _fake_embed("事件")
        m1 = _make_mem("episodic", "事件 A", embedding=emb, decay_weight=0.5)
        m2 = _make_mem("episodic", "事件 A", embedding=emb)

        await write_memories([m1])
        report = await write_memories([m2])
        assert report.inserted == 0
        assert report.reinforced == 1

        active = list_active_memories()
        assert len(active) == 1
        # 强化 = 提权重 + 计访问
        assert active[0].decay_weight == pytest.approx(0.5 + EPISODIC_REINFORCE_BOOST)
        assert active[0].access_count == 1

    @pytest.mark.asyncio
    async def test_episodic_distinct_events_both_inserted(self):
        """相似度低于阈值的两件事各自成条——门槛只拦真重复。"""
        m1 = _make_mem("episodic", "事件 A", embedding=_fake_embed("事件 A"))
        m2 = _make_mem("episodic", "事件 B", embedding=_fake_embed("完全不同的另一件事"))

        await write_memories([m1])
        report = await write_memories([m2])
        assert report.inserted == 1
        assert report.reinforced == 0
        assert len(list_active_memories()) == 2

    @pytest.mark.asyncio
    async def test_episodic_dedup_can_be_disabled(self, monkeypatch):
        """EPISODIC_DEDUP_ENABLED=false → 回到无条件直写（Phase 4 等价）。"""
        import config

        monkeypatch.setattr(config, "EPISODIC_DEDUP_ENABLED", False)
        emb = _fake_embed("事件")
        await write_memories([_make_mem("episodic", "事件 A", embedding=emb)])
        report = await write_memories([_make_mem("episodic", "事件 A", embedding=emb)])
        assert report.inserted == 1
        assert len(list_active_memories()) == 2

    @pytest.mark.asyncio
    async def test_episodic_without_embedding_inserts(self):
        """无 embedding 无法比相似度 → 直写，不因此丢数据。"""
        report = await write_memories([_make_mem("episodic", "无向量事件")])
        assert report.inserted == 1


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
    async def test_semantic_override_with_identical_embedding(self, monkeypatch):
        """相同 embedding → 必定触发覆盖（相似度 1.0）。旧阈值逻辑：关闭冲突判定。"""
        monkeypatch.setattr(config, "CONFLICT_DETECT_ENABLED", False)
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
    async def test_procedural_dedup_on_identical_embedding(self, monkeypatch):
        """相同 embedding → 去重，不 insert，mark_accessed 旧记忆。旧阈值逻辑：关闭冲突判定。"""
        monkeypatch.setattr(config, "CONFLICT_DETECT_ENABLED", False)
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
        good = _make_mem("episodic", "正常记忆", embedding=_fake_embed("正常"))
        dup_id = str(uuid.uuid4())

        # 先写入一条记忆占据该 ID。
        # 三条记忆的向量两两都不相似——否则会先命中 P4-① 的近重复强化，
        # 走不到 insert，也就触发不了这里要测的主键冲突。
        existing = Memory(
            memory_id=dup_id,
            type="episodic",
            content="先占据",
            embedding=_fake_embed("先占据"),
        )
        insert_memory(existing)

        # 构造一条会触发 UNIQUE 约束冲突的记忆（相同 memory_id）
        bad = Memory(
            memory_id=dup_id,  # 重复 ID → 触发 PRIMARY KEY 冲突
            type="episodic",
            content="重复ID记忆",
            embedding=_fake_embed("重复ID记忆"),
        )

        report = await write_memories([bad, good])
        assert report.failed >= 1
        # 正常记忆仍被写入
        assert report.inserted >= 1


# ═══════════════════════════════════════════════════════════
# 冲突感知写入管线（CONFLICT_DETECT_ENABLED=True，默认）
# judge_relations / embed_text 均 mock，不发真实请求
# ═══════════════════════════════════════════════════════════

EMB = _fake_embed("冲突测试共用向量")  # 共用同一向量 → 相似度 1.0，必入候选


def _judge(monkeypatch, relation=None, *, plan=None, subtype=None, merged=None,
           judged_by="llm", reason="理由"):
    """mock judge_relations。relation 对所有候选生效；plan(cand)->(relation, subtype, merged) 逐对指定。"""
    calls: list[list] = []

    async def fake(cands):
        calls.append(list(cands))
        out = []
        for c in cands:
            rel, sub, mg = plan(c) if plan else (relation, subtype, merged)
            out.append(Relation(
                new_id=c.new.memory_id, old_id=c.old.memory_id, relation=rel,
                similarity=c.similarity, judged_by=judged_by, subtype=sub,
                reason=reason, merged=mg,
            ))
        return out

    monkeypatch.setattr(mw, "judge_relations", fake)
    return calls


def _mock_embed(monkeypatch, fail=False):
    async def fake(text):
        if fail:
            raise RuntimeError("embedding down")
        return _fake_embed(text)

    monkeypatch.setattr(mw, "embed_text", fake)


def _seed(mem_type, content, **kw):
    mem = _make_mem(mem_type, content, embedding=EMB, **kw)
    insert_memory(mem)
    return mem


def _new(mem_type, content, **kw):
    return _make_mem(mem_type, content, embedding=EMB, **kw)


def _archived_ids():
    return {m.memory_id for m in list_archived_memories()}


def _active_ids():
    return {m.memory_id for m in list_active_memories()}


class TestRelationMatrix:
    """关系 × 类型处置矩阵，逐格断言热表/归档表/关系表/decay_weight/unresolved。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mem_type,relation,subtype,old_state,link,fields",
        [
            ("semantic", "update", None, "archived", ("update", "resolved", None),
             dict(inserted=1, updated=1, superseded=1)),
            ("semantic", "contradict", None, "active", ("contradict", "pending", None),
             dict(inserted=1, conflicted=1)),
            ("semantic", "refine", None, "active", ("refine", "resolved", None),
             dict(inserted=1, refined=1)),
            ("procedural", "update", None, "archived", ("update", "resolved", None),
             dict(inserted=1, updated=1, superseded=1)),
            ("procedural", "contradict", None, "archived", ("update", "resolved", None),
             dict(inserted=1, updated=1, superseded=1)),
            ("procedural", "refine", None, "active", ("refine", "resolved", None),
             dict(inserted=1, refined=1)),
            ("episodic", "update", "correction", "archived", ("update", "resolved", "correction"),
             dict(inserted=1, updated=1, superseded=1)),
            ("episodic", "update", "followup", "active", ("update", "resolved", "followup"),
             dict(inserted=1, updated=1)),
            ("episodic", "contradict", None, "active", ("contradict", "pending", None),
             dict(inserted=1, conflicted=1)),
            ("episodic", "refine", None, "active", ("refine", "resolved", None),
             dict(inserted=1, refined=1)),
        ],
    )
    async def test_cell(self, monkeypatch, mem_type, relation, subtype, old_state, link, fields):
        old = _seed(mem_type, "旧内容", unresolved=(mem_type == "episodic"))
        _judge(monkeypatch, relation, subtype=subtype)
        new = _new(mem_type, "新内容")
        report = await write_memories([new])

        for name, val in fields.items():
            assert getattr(report, name) == val, name
        assert report.failed == 0

        assert new.memory_id in _active_ids()
        if old_state == "archived":
            assert old.memory_id not in _active_ids()
            assert old.memory_id in _archived_ids()
        else:
            assert old.memory_id in _active_ids()
            assert get_memory(old.memory_id).superseded_by is None
        if old_state == "archived":
            # 归档前已标记 superseded_by
            arch = [m for m in list_archived_memories() if m.memory_id == old.memory_id][0]
            assert arch.superseded_by == new.memory_id

        links = links_for([old.memory_id])
        assert len(links) == 1
        l = links[0]
        assert (l.relation, l.status, l.subtype) == link
        assert (l.src_id, l.dst_id) == (new.memory_id, old.memory_id)
        assert l.judged_by == "llm"

        # unresolved：只有 followup 清除
        if mem_type == "episodic":
            cleared = not get_memory(old.memory_id).unresolved if old_state == "active" else None
            assert cleared == (True if subtype == "followup" else False if old_state == "active" else None)
        # decay_weight 不动
        if old_state == "active":
            assert get_memory(old.memory_id).decay_weight == pytest.approx(1.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("relation", ["update", "contradict"])
    async def test_emotional_flip_damps_old(self, monkeypatch, relation):
        old = _seed("emotional", "旧态度", valence=0.8)
        _judge(monkeypatch, relation)
        new = _new("emotional", "新态度", valence=-0.8)
        report = await write_memories([new])
        assert report.inserted == 1 and report.damped == 1
        assert report.superseded == 0
        assert old.memory_id in _active_ids()
        assert get_memory(old.memory_id).decay_weight == pytest.approx(
            config.EMOTIONAL_FLIP_DAMPING
        )
        (l,) = links_for([old.memory_id])
        assert (l.relation, l.status, l.subtype) == ("update", "resolved", "flip")

    @pytest.mark.asyncio
    async def test_emotional_flip_ignores_guards(self, monkeypatch):
        """emotional 不走守卫：旧记忆是 remember 且高 access 也照样降权。"""
        old = _seed("emotional", "旧态度", origin="remember", access_count=99)
        _judge(monkeypatch, "update")
        report = await write_memories([_new("emotional", "新态度")])
        assert report.damped == 1 and report.conflicted == 0

    @pytest.mark.asyncio
    async def test_emotional_refine_reinforces_without_insert(self, monkeypatch):
        old = _seed("emotional", "旧态度", arousal=0.5, tags=["a"], decay_weight=0.6)
        _judge(monkeypatch, "refine")
        report = await write_memories([_new("emotional", "补充", arousal=0.9, tags=["b"])])
        assert report.inserted == 0
        assert report.reinforced == 1 and report.refined == 1
        got = get_memory(old.memory_id)
        assert got.decay_weight == pytest.approx(0.6 + config.EMOTIONAL_REINFORCE_BOOST)
        assert got.arousal == 0.9 and set(got.tags) == {"a", "b"}
        assert links_for([old.memory_id]) == []
        assert len(list_active_memories()) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mem_type", ["semantic", "procedural"])
    async def test_duplicate_semantic_procedural(self, monkeypatch, mem_type):
        old = _seed(mem_type, "旧内容")
        _judge(monkeypatch, "duplicate")
        report = await write_memories([_new(mem_type, "新内容")])
        assert report.inserted == 0 and report.deduped == 1
        assert get_memory(old.memory_id).access_count == 1
        assert len(list_active_memories()) == 1
        assert links_for([old.memory_id]) == []

    @pytest.mark.asyncio
    async def test_duplicate_emotional_episodic_reinforce(self, monkeypatch):
        e_old = _seed("emotional", "旧情绪", decay_weight=0.5, arousal=0.4)
        _judge(monkeypatch, "duplicate")
        r = await write_memories([_new("emotional", "新情绪", arousal=0.7)])
        assert r.inserted == 0 and r.reinforced == 1
        got = get_memory(e_old.memory_id)
        assert got.decay_weight == pytest.approx(0.5 + config.EMOTIONAL_REINFORCE_BOOST)
        assert got.arousal == 0.7

        p_old = _seed("episodic", "旧事件", decay_weight=0.5)
        r = await write_memories([_new("episodic", "新事件")])
        assert r.inserted == 0 and r.reinforced == 1
        assert get_memory(p_old.memory_id).decay_weight == pytest.approx(
            0.5 + EPISODIC_REINFORCE_BOOST
        )

    @pytest.mark.asyncio
    async def test_unrelated_inserts_without_link(self, monkeypatch):
        old = _seed("semantic", "旧内容")
        _judge(monkeypatch, "unrelated")
        new = _new("semantic", "新内容")
        report = await write_memories([new])
        assert report.inserted == 1
        assert _active_ids() == {old.memory_id, new.memory_id}
        assert links_for([old.memory_id]) == []
        assert any(line.startswith("[unrelated|") for line in mw.LAST_JUDGMENTS)

    @pytest.mark.asyncio
    async def test_cross_type_episodic_new_overrides_semantic(self, monkeypatch):
        old = _seed("semantic", "用户住在北京")
        _judge(monkeypatch, "update")
        new = _new("episodic", "上周搬到了上海")
        report = await write_memories([new])
        assert report.inserted == 1 and report.updated == 1
        assert old.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_cross_type_semantic_new_leaves_episodic(self, monkeypatch):
        old = _seed("episodic", "上周去了北京出差")
        _judge(monkeypatch, "update")
        new = _new("semantic", "用户住在上海")
        report = await write_memories([new])
        assert report.inserted == 1
        assert report.superseded == 0
        assert old.memory_id in _active_ids()
        (l,) = links_for([old.memory_id])
        assert (l.relation, l.status) == ("update", "resolved")


class TestRefineMerge:
    @pytest.mark.asyncio
    async def test_merged_replaces_new_and_supersedes_old(self, monkeypatch):
        _mock_embed(monkeypatch)
        old = _seed("semantic", "用户是后端开发")
        _judge(monkeypatch, "refine", merged="用户是后端开发，主要写 Go")
        new = _new("semantic", "用户主要写 Go", tags=["t"])
        report = await write_memories([new])
        assert report.inserted == 1 and report.refined == 1 and report.superseded == 1

        active = list_active_memories()
        assert len(active) == 1
        merged = active[0]
        assert merged.content == "用户是后端开发，主要写 Go"
        assert merged.memory_id != new.memory_id
        assert merged.tags == ["t"]
        assert np.allclose(merged.embedding, _fake_embed(merged.content))
        assert old.memory_id in _archived_ids()
        (l,) = links_for([old.memory_id])
        assert (l.src_id, l.relation, l.status) == (merged.memory_id, "refine", "resolved")

    @pytest.mark.asyncio
    async def test_merged_guard_failure_inserts_original_and_pends(self, monkeypatch):
        _mock_embed(monkeypatch)
        old = _seed("semantic", "用户是后端开发", access_count=10)
        _judge(monkeypatch, "refine", merged="合并文本")
        new = _new("semantic", "用户主要写 Go")
        report = await write_memories([new])
        assert report.inserted == 1 and report.conflicted == 1
        assert _active_ids() == {old.memory_id, new.memory_id}
        (l,) = links_for([old.memory_id])
        assert (l.relation, l.status) == ("contradict", "pending")

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_original(self, monkeypatch):
        _mock_embed(monkeypatch, fail=True)
        old = _seed("semantic", "用户是后端开发")
        _judge(monkeypatch, "refine", merged="合并文本")
        new = _new("semantic", "用户主要写 Go")
        report = await write_memories([new])
        assert report.inserted == 1 and report.refined == 1 and report.superseded == 0
        assert _active_ids() == {old.memory_id, new.memory_id}
        (l,) = links_for([old.memory_id])
        assert (l.relation, l.status) == ("refine", "resolved")

    @pytest.mark.asyncio
    async def test_refine_without_merged_inserts_original(self, monkeypatch):
        old = _seed("procedural", "回复要简短")
        _judge(monkeypatch, "refine", merged=None)
        new = _new("procedural", "回复要简短且带结论")
        await write_memories([new])
        assert _active_ids() == {old.memory_id, new.memory_id}


class TestBatch:
    @pytest.mark.asyncio
    async def test_second_new_sees_first_new(self, monkeypatch):
        """同批次两条矛盾新记忆：第二条能看到第一条，整批只调一次判定。"""
        calls = _judge(monkeypatch, "contradict")
        a = _new("semantic", "用户喝咖啡")
        b = _new("semantic", "用户不喝咖啡")
        report = await write_memories([a, b])
        assert len(calls) == 1 and len(calls[0]) == 1
        assert calls[0][0].new.memory_id == b.memory_id
        assert calls[0][0].old.memory_id == a.memory_id
        assert report.inserted == 2 and report.conflicted == 1
        (l,) = list_pending_links()
        assert (l.src_id, l.dst_id) == (b.memory_id, a.memory_id)

    @pytest.mark.asyncio
    async def test_batch_update_archives_earlier_new(self, monkeypatch):
        _judge(monkeypatch, "update")
        a = _new("semantic", "用户住在北京")
        b = _new("semantic", "用户搬到了上海")
        report = await write_memories([a, b])
        assert report.inserted == 2 and report.superseded == 1
        assert _active_ids() == {b.memory_id}
        assert a.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_absorbed_duplicate_relations_are_ignored(self, monkeypatch):
        """n2 被判 duplicate 吸收后，n3 对 n2 的关系被忽略，只处理 n3 对 n1 的。"""
        n1 = _new("semantic", "事实一")
        n2 = _new("semantic", "事实二")
        n3 = _new("semantic", "事实三")

        def plan(c):
            if c.new.memory_id == n2.memory_id:
                return ("duplicate", None, None)
            if c.old.memory_id == n2.memory_id:
                return ("update", None, None)
            return ("contradict", None, None)

        _judge(monkeypatch, plan=plan)
        report = await write_memories([n1, n2, n3])
        assert report.inserted == 2 and report.deduped == 1
        assert _active_ids() == {n1.memory_id, n3.memory_id}
        links = list_pending_links()
        assert len(links) == 1
        assert (links[0].src_id, links[0].dst_id) == (n3.memory_id, n1.memory_id)
        assert links_for([n2.memory_id]) == []

    @pytest.mark.asyncio
    async def test_old_already_archived_in_batch_is_ignored(self, monkeypatch):
        """同一旧记忆被前一条新记忆覆盖归档后，后一条对它的关系忽略。"""
        old = _seed("semantic", "旧事实")
        a = _new("semantic", "新事实A")
        b = _new("semantic", "新事实B")

        def plan(c):
            if c.old.memory_id == old.memory_id:
                return ("update", None, None)
            return ("unrelated", None, None)

        _judge(monkeypatch, plan=plan)
        report = await write_memories([a, b])
        assert report.inserted == 2
        assert report.updated == 1 and report.superseded == 1
        assert len([l for l in links_for([old.memory_id])]) == 1

    @pytest.mark.asyncio
    async def test_duplicate_wins_over_other_relations(self, monkeypatch):
        old1 = _seed("semantic", "旧一")
        old2 = _seed("semantic", "旧二")

        def plan(c):
            return ("duplicate", None, None) if c.old.memory_id == old1.memory_id \
                else ("update", None, None)

        _judge(monkeypatch, plan=plan)
        report = await write_memories([_new("semantic", "新")])
        assert report.inserted == 0 and report.deduped == 1 and report.updated == 0
        assert _active_ids() == {old1.memory_id, old2.memory_id}


class TestGuards:
    @pytest.mark.asyncio
    async def test_guard_time_older_new_does_not_override(self, monkeypatch):
        """旧日记导入：新记忆 created_at 更早，不覆盖较新的事实。"""
        old = _seed("semantic", "用户住在上海", created_at="2026-11-01 00:00:00")
        _judge(monkeypatch, "update", judged_by="rule")
        new = _new("semantic", "用户住在北京", created_at="2026-01-01 00:00:00", origin="import")
        report = await write_memories([new])
        assert report.inserted == 1 and report.conflicted == 1 and report.updated == 0
        assert _active_ids() == {old.memory_id, new.memory_id}
        (l,) = list_pending_links()
        assert (l.relation, l.status) == ("update", "pending")
        assert any(line.startswith("[update→pending|") and "早于" in line
                   for line in mw.LAST_JUDGMENTS)

    @pytest.mark.asyncio
    async def test_guard_time_empty_created_at_counts_as_now(self, monkeypatch):
        old = _seed("semantic", "旧", created_at="2026-11-01 00:00:00")
        _judge(monkeypatch, "update")
        report = await write_memories([_new("semantic", "新")])
        assert report.updated == 1 and old.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_guard_origin_remember_protected_from_dehydrate(self, monkeypatch):
        old = _seed("semantic", "用户住在北京", origin="remember")
        _judge(monkeypatch, "update")
        new = _new("semantic", "用户住在上海", origin="dehydrate")
        report = await write_memories([new])
        assert report.conflicted == 1 and report.updated == 0
        assert _active_ids() == {old.memory_id, new.memory_id}
        assert any("remember" in line for line in mw.LAST_JUDGMENTS)

    @pytest.mark.asyncio
    async def test_guard_origin_remember_can_override_remember(self, monkeypatch):
        old = _seed("semantic", "用户住在北京", origin="remember")
        _judge(monkeypatch, "update")
        report = await write_memories([_new("semantic", "用户住在上海", origin="remember")])
        assert report.updated == 1 and old.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_guard_access_protects_semantic(self, monkeypatch):
        old = _seed("semantic", "用户住在北京", access_count=config.CONFLICT_PROTECT_ACCESS)
        _judge(monkeypatch, "update")
        new = _new("semantic", "用户住在上海")
        report = await write_memories([new])
        assert report.conflicted == 1 and report.updated == 0
        assert _active_ids() == {old.memory_id, new.memory_id}
        assert any("access_count" in line for line in mw.LAST_JUDGMENTS)

    @pytest.mark.asyncio
    async def test_procedural_not_protected_by_access(self, monkeypatch):
        old = _seed("procedural", "回复要长", access_count=99)
        _judge(monkeypatch, "update")
        report = await write_memories([_new("procedural", "回复要短")])
        assert report.updated == 1 and report.conflicted == 0
        assert old.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_procedural_time_guard_keeps_old_and_pends(self, monkeypatch):
        old = _seed("procedural", "回复要长", created_at="2026-11-01 00:00:00")
        _judge(monkeypatch, "update")
        new = _new("procedural", "回复要短", created_at="2026-01-01 00:00:00")
        report = await write_memories([new])
        assert report.inserted == 1 and report.conflicted == 1
        assert _active_ids() == {old.memory_id, new.memory_id}
        (l,) = list_pending_links()
        assert l.status == "pending"

    @pytest.mark.asyncio
    async def test_episodic_correction_is_guarded_but_followup_is_not(self, monkeypatch):
        old = _seed("episodic", "会议在周五", access_count=99, unresolved=True)
        _judge(monkeypatch, "update", subtype="correction")
        r = await write_memories([_new("episodic", "会议改到周三")])
        assert r.conflicted == 1 and old.memory_id in _active_ids()

        _judge(monkeypatch, "update", subtype="followup")
        r = await write_memories([_new("episodic", "会议开完了")])
        assert r.updated >= 1
        assert get_memory(old.memory_id).unresolved is False


class TestFallbacks:
    @pytest.mark.asyncio
    async def test_judge_exception_falls_back_to_rules_and_keeps_new(self, monkeypatch):
        async def boom(cands):
            raise RuntimeError("llm down")

        monkeypatch.setattr(mw, "judge_relations", boom)
        old = _seed("procedural", "回复要长")
        new = _new("procedural", "回复要短")
        report = await write_memories([new])
        assert report.failed == 0
        assert new.memory_id in _active_ids()  # 规则：procedural 过阈值 → update
        assert old.memory_id in _archived_ids()

    @pytest.mark.asyncio
    async def test_judge_wrong_length_falls_back_to_rules(self, monkeypatch):
        async def short(cands):
            return []

        monkeypatch.setattr(mw, "judge_relations", short)
        _seed("emotional", "喜欢加班", valence=0.7)
        new = _new("emotional", "讨厌加班", valence=-0.7)
        report = await write_memories([new])
        assert report.inserted == 1 and report.damped == 1

    @pytest.mark.asyncio
    async def test_no_candidates_means_no_judge_call(self, monkeypatch):
        calls = _judge(monkeypatch, "update")
        report = await write_memories([
            _new("semantic", "A"),
            _make_mem("semantic", "无向量", embedding=None),
            _make_mem("semantic", "低相似", embedding=-EMB),
        ])
        assert report.inserted == 3
        assert calls == []

    @pytest.mark.asyncio
    async def test_new_without_embedding_not_recalled(self, monkeypatch):
        old = _seed("semantic", "旧")
        calls = _judge(monkeypatch, "update")
        report = await write_memories([_make_mem("semantic", "无向量", embedding=None)])
        assert report.inserted == 1 and calls == []
        assert old.memory_id in _active_ids()

    @pytest.mark.asyncio
    async def test_single_failure_does_not_block_others(self, monkeypatch):
        _judge(monkeypatch, "update")
        _seed("semantic", "旧")
        real = mw.supersede_and_archive
        state = {"n": 0}

        def flaky(ids, new_id):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("db hiccup")
            return real(ids, new_id)

        monkeypatch.setattr(mw, "supersede_and_archive", flaky)
        a = _new("semantic", "A")
        report = await write_memories([a])
        assert report.failed == 1


class TestSwitchOff:
    @pytest.mark.asyncio
    async def test_detect_disabled_uses_legacy_logic_and_no_judge(self, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_DETECT_ENABLED", False)
        calls = _judge(monkeypatch, "contradict")
        old = _seed("semantic", "旧")
        new = _new("semantic", "新")
        report = await write_memories([new])
        assert calls == []
        assert report.inserted == 1 and report.superseded == 1
        assert report.updated == report.conflicted == report.refined == report.damped == 0
        got = get_memory(old.memory_id)
        assert got.superseded_by == new.memory_id
        assert old.memory_id not in _archived_ids()  # 旧逻辑只标记，不归档
        assert list_pending_links() == []
        assert mw.LAST_JUDGMENTS == []

    @pytest.mark.asyncio
    async def test_detect_disabled_procedural_still_dedups(self, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_DETECT_ENABLED", False)
        old = _seed("procedural", "回复要长")
        report = await write_memories([_new("procedural", "回复要短")])
        assert report.inserted == 0 and report.deduped == 1
        assert get_memory(old.memory_id).access_count == 1


class TestLastJudgments:
    @pytest.mark.asyncio
    async def test_format_update(self, monkeypatch):
        _seed("semantic", "用户住在北京")
        _judge(monkeypatch, "update", reason="居住地发生变化")
        await write_memories([_new("semantic", "用户搬到了上海")])
        assert mw.LAST_JUDGMENTS == [
            "[update|sim=1.00|llm] 用户搬到了上海 → 覆盖：用户住在北京（居住地发生变化）"
        ]

    @pytest.mark.asyncio
    async def test_format_contradict_and_pending_and_reset(self, monkeypatch):
        _seed("semantic", "用户每天喝咖啡")
        _judge(monkeypatch, "contradict", reason="无时间线索")
        await write_memories([_new("semantic", "用户不喝咖啡")])
        assert mw.LAST_JUDGMENTS == [
            "[contradict|sim=1.00|llm] 用户不喝咖啡 ⇄ 挂起：用户每天喝咖啡（无时间线索）"
        ]

        # 下一次写入开头清空
        _seed("semantic", "用户住在北京", access_count=7)
        _judge(monkeypatch, "update", judged_by="rule", reason="")
        await write_memories([_new("semantic", "用户搬到了上海")])
        lines = [l for l in mw.LAST_JUDGMENTS]
        assert not any(l.startswith("[contradict|") for l in lines)
        assert any(l.startswith("[update→pending|sim=1.00|rule] 用户搬到了上海 ⇄ 挂起：用户住在北京（")
                   and "access_count=7" in l for l in lines)

    @pytest.mark.asyncio
    async def test_duplicate_logged_and_empty_write_clears(self, monkeypatch):
        _seed("semantic", "旧")
        _judge(monkeypatch, "duplicate")
        await write_memories([_new("semantic", "新")])
        assert len(mw.LAST_JUDGMENTS) == 1
        assert mw.LAST_JUDGMENTS[0].startswith("[duplicate|")
        await write_memories([])
        assert mw.LAST_JUDGMENTS == []
