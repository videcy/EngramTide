"""
测试 SQLite 存储层：初始化、插入、读取、embedding 序列化、访问统计。
"""

import uuid

import numpy as np
import pytest

from core.memory_store import (
    Memory,
    close_db,
    init_db,
    insert_memory,
    insert_memories,
    list_active_memories,
    list_recent_memories,
    mark_accessed,
)


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """每个测试使用独立临时数据库。"""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)

    # 重置连接
    import core.memory_store as ms

    ms._connection = None
    init_db()
    yield
    close_db()


# ── 测试：初始化 ──────────────────────────────────────────


def test_init_db_creates_table():
    """初始化后，表应该存在。"""
    import sqlite3

    from core.memory_store import _get_conn

    conn = _get_conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories';"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "memories"


def test_init_db_idempotent():
    """重复 init_db() 不报错。"""
    init_db()
    init_db()  # 不应抛出异常


# ── 测试：插入与读取 ──────────────────────────────────────


def test_insert_and_read_one_memory():
    """插入一条记忆后可读取。"""
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        type="semantic",
        content="用户在做 AI 记忆系统项目",
    )
    insert_memory(mem)

    all_mem = list_active_memories()
    assert len(all_mem) == 1
    assert all_mem[0].content == mem.content
    assert all_mem[0].type == "semantic"


def test_insert_preserves_access_count():
    """回归（Phase 4 合并 bug）：insert_memory 必须落库 access_count，
    否则合并记忆的访问历史归零、衰减减速因子失效。"""
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        type="episodic",
        content="带访问历史的记忆",
        access_count=7,
    )
    insert_memory(mem)

    loaded = list_active_memories()[0]
    assert loaded.access_count == 7

    # 带显式时间戳的插入路径（INSERT_WITH_TIME_SQL）同样保持
    mem2 = Memory(
        memory_id=str(uuid.uuid4()),
        type="episodic",
        content="带时间戳与访问历史的记忆",
        access_count=5,
        created_at="2026-01-01 00:00:00",
    )
    insert_memory(mem2)
    loaded2 = [m for m in list_active_memories() if m.memory_id == mem2.memory_id][0]
    assert loaded2.access_count == 5


def test_insert_memories_batch():
    """批量插入多条记忆。"""
    mems = [
        Memory(memory_id=str(uuid.uuid4()), type="semantic", content=f"记忆 #{i}")
        for i in range(3)
    ]
    insert_memories(mems)

    all_mem = list_active_memories()
    assert len(all_mem) == 3


def test_list_recent_respects_limit():
    """list_recent_memories 限制返回数量。"""
    for i in range(15):
        insert_memory(
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content=f"记忆 #{i:02d}",
            )
        )

    recent = list_recent_memories(limit=5)
    assert len(recent) == 5


# ── 测试：embedding 序列化 ─────────────────────────────────


def test_embedding_roundtrip():
    """embedding 写入和读取后向量数值一致。"""
    original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        type="semantic",
        content="测试 embedding",
        embedding=original,
    )
    insert_memory(mem)

    loaded = list_active_memories()[0]
    assert loaded.embedding is not None
    np.testing.assert_array_equal(loaded.embedding, original)


def test_null_embedding_roundtrip():
    """无 embedding 的记忆读取不崩溃。"""
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        type="semantic",
        content="无 embedding",
        embedding=None,
    )
    insert_memory(mem)

    loaded = list_active_memories()[0]
    assert loaded.embedding is None


# ── 测试：访问统计 ────────────────────────────────────────


def test_mark_accessed_updates_stats():
    """mark_accessed 更新 access_count 和 last_accessed。"""
    mid = str(uuid.uuid4())
    insert_memory(Memory(memory_id=mid, type="semantic", content="测试访问"))

    mark_accessed([mid])

    all_mem = list_active_memories()
    assert all_mem[0].access_count == 1
    assert all_mem[0].last_accessed != ""


def test_mark_accessed_empty_list_no_error():
    """空列表不报错。"""
    mark_accessed([])


# ── 测试：superseded_by 过滤 ───────────────────────────────


def test_superseded_memory_excluded():
    """被取代的记忆不出现在 active 列表中。"""
    mid_active = str(uuid.uuid4())
    mid_old = str(uuid.uuid4())

    insert_memory(Memory(memory_id=mid_active, type="semantic", content="活跃"))
    insert_memory(
        Memory(
            memory_id=mid_old,
            type="semantic",
            content="已取代",
            superseded_by=mid_active,
        )
    )

    active = list_active_memories()
    assert len(active) == 1
    assert active[0].memory_id == mid_active


# ── Phase 2 测试：update_decay_weights ─────────────────────


def test_update_decay_weights_batch():
    """批量更新 decay_weight 后数值一致。"""
    mid1 = str(uuid.uuid4())
    mid2 = str(uuid.uuid4())
    insert_memory(Memory(memory_id=mid1, type="episodic", content="事件1", decay_weight=1.0))
    insert_memory(Memory(memory_id=mid2, type="episodic", content="事件2", decay_weight=1.0))

    from core.memory_store import update_decay_weights

    # update_decay_weights expects (new_weight, memory_id) tuples
    update_decay_weights([(0.5, mid1), (0.3, mid2)])

    active = {m.memory_id: m for m in list_active_memories()}
    assert active[mid1].decay_weight == 0.5
    assert active[mid2].decay_weight == 0.3


def test_update_decay_weights_empty_list_no_error():
    """空列表不报错。"""
    from core.memory_store import update_decay_weights

    update_decay_weights([])


# ── Phase 2 测试：mark_superseded ──────────────────────────


def test_mark_superseded_batch():
    """批量标记 superseded 后旧记忆从 active 消失。"""
    new_id = str(uuid.uuid4())
    old1 = str(uuid.uuid4())
    old2 = str(uuid.uuid4())

    insert_memory(Memory(memory_id=old1, type="semantic", content="旧1"))
    insert_memory(Memory(memory_id=old2, type="semantic", content="旧2"))

    from core.memory_store import mark_superseded

    mark_superseded([old1, old2], new_id)

    # 全部被排除
    active = list_active_memories()
    assert len(active) == 0


def test_mark_superseded_empty_list_no_error():
    """空列表直接返回。"""
    from core.memory_store import mark_superseded

    mark_superseded([], "any-id")


# ── Phase 2 测试：reinforce_memory ─────────────────────────


def test_reinforce_memory_updates_fields():
    """强化更新：decay_weight 提升、arousal 取 max、tags 合并、access_count +1。"""
    mid = str(uuid.uuid4())
    insert_memory(
        Memory(
            memory_id=mid,
            type="emotional",
            content="讨厌画大饼",
            decay_weight=0.6,
            arousal=0.5,
            tags=["职场"],
            access_count=0,
        )
    )

    from core.memory_store import reinforce_memory

    reinforce_memory(mid, 0.8, 0.9, ["职场", "吐槽"])

    mems = {m.memory_id: m for m in list_active_memories()}
    updated = mems[mid]
    assert updated.decay_weight == 0.8
    assert updated.arousal == 0.9
    assert set(updated.tags) == {"职场", "吐槽"}
    assert updated.access_count == 1


# ── Phase 2 测试：get_meta / set_meta ──────────────────────


def test_get_meta_nonexistent_returns_none():
    """不存在的 key 返回 None。"""
    from core.memory_store import get_meta

    assert get_meta("nonexistent_key") is None


def test_set_meta_and_get_meta_roundtrip():
    """set_meta 写入后 get_meta 可读取。"""
    from core.memory_store import get_meta, set_meta

    set_meta("test_key", "test_value")
    assert get_meta("test_key") == "test_value"


def test_set_meta_overwrites_existing():
    """二次写入是覆盖而非报错。"""
    from core.memory_store import get_meta, set_meta

    set_meta("dup_key", "v1")
    set_meta("dup_key", "v2")
    assert get_meta("dup_key") == "v2"


# ── Phase 2 测试：指定时间插入 ──────────────────────────────


def test_insert_with_specified_created_at():
    """指定 created_at 插入后读回一致。"""
    mid = str(uuid.uuid4())
    insert_memory(
        Memory(
            memory_id=mid,
            type="episodic",
            content="指定时间事件",
            created_at="2026-01-15 10:30:00",
        )
    )

    mems = {m.memory_id: m for m in list_active_memories()}
    assert "2026-01-15" in mems[mid].created_at


def test_insert_without_created_at_uses_default():
    """不指定 created_at 时 DB 默认 CURRENT_TIMESTAMP 生效。"""
    mid = str(uuid.uuid4())
    insert_memory(Memory(memory_id=mid, type="semantic", content="默认时间"))

    mems = list_active_memories()
    assert len(mems) == 1
    # created_at 应该有值（DB 默认生成的）
    assert mems[0].created_at != ""


# ── Phase 2 测试：init_db 新增索引与 meta 表 ───────────────


def test_init_db_creates_meta_table():
    """init_db 后 meta 表存在。"""
    import sqlite3

    from core.memory_store import _get_conn

    conn = _get_conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta';"
    ).fetchall()
    assert len(rows) == 1


def test_init_db_keeps_only_the_index_that_is_actually_used():
    """
    P1：唯一的热查询是 `WHERE superseded_by IS NULL ORDER BY created_at DESC`。

    原先那四条索引对它一条都用不上（仍是全表扫 + 排序），只在写入时增加 B 树
    维护成本，因此全部删除，换成同时覆盖过滤条件与排序键的部分索引。
    """
    from core.memory_store import _get_conn

    conn = _get_conn()
    index_names = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index';"
        ).fetchall()
    }
    assert "idx_active_created" in index_names
    for legacy in (
        "idx_memories_type",
        "idx_memories_decay",
        "idx_memories_created",
        "idx_memories_unresolved",
    ):
        assert legacy not in index_names

    # 部分索引确实被查询计划用上（不是建了个摆设）
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT memory_id FROM memories "
        "WHERE superseded_by IS NULL ORDER BY created_at DESC;"
    ).fetchall()
    assert any("idx_active_created" in str(tuple(row)) for row in plan)


# ── 冲突处理：origin / memory_links / 覆盖归档 ─────────────


def _mk(content="用户住在北京", mtype="semantic", **kw):
    mid = kw.pop("memory_id", str(uuid.uuid4()))
    m = Memory(memory_id=mid, type=mtype, content=content, **kw)
    insert_memory(m)
    return mid


def test_origin_roundtrip_insert_archive_restore():
    import core.memory_store as ms

    a = _mk(origin="remember")
    b = str(uuid.uuid4())
    insert_memories([Memory(memory_id=b, type="semantic", content="批量", origin="import")])
    c = _mk("默认来源")
    assert ms.get_memory(a).origin == "remember"
    assert ms.get_memory(b).origin == "import"
    assert ms.get_memory(c).origin == "dehydrate"

    ms.archive_memories([a])
    assert [m.origin for m in ms.list_archived_memories()] == ["remember"]
    ms.restore_archived([a])
    assert ms.get_memory(a).origin == "remember"


def test_migrate_old_db_without_origin_and_links(monkeypatch, tmp_path):
    import sqlite3

    import core.memory_store as ms

    close_db()
    db_file = tmp_path / "old.db"
    raw = sqlite3.connect(db_file)
    raw.execute(
        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, type TEXT NOT NULL, "
        "content TEXT NOT NULL, valence REAL DEFAULT 0.0, arousal REAL DEFAULT 0.0, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP, access_count INTEGER DEFAULT 0, "
        "decay_weight REAL DEFAULT 1.0, embedding BLOB, source_conv_id TEXT, "
        "unresolved BOOLEAN DEFAULT FALSE, tags TEXT DEFAULT '', superseded_by TEXT DEFAULT NULL)"
    )
    raw.execute(
        "CREATE TABLE memories_archive (memory_id TEXT PRIMARY KEY, type TEXT NOT NULL, "
        "content TEXT NOT NULL, valence REAL DEFAULT 0.0, arousal REAL DEFAULT 0.0, "
        "created_at TIMESTAMP, last_accessed TIMESTAMP, access_count INTEGER DEFAULT 0, "
        "decay_weight REAL DEFAULT 1.0, embedding BLOB, source_conv_id TEXT, "
        "unresolved BOOLEAN DEFAULT FALSE, tags TEXT DEFAULT '', "
        "superseded_by TEXT DEFAULT NULL, archived_at TEXT NOT NULL)"
    )
    raw.execute("INSERT INTO memories (memory_id, type, content) VALUES ('o1','semantic','旧库记忆')")
    raw.execute(
        "INSERT INTO memories_archive (memory_id, type, content, archived_at) "
        "VALUES ('o2','episodic','旧归档','2026-01-01T00:00:00')"
    )
    raw.commit()
    raw.close()

    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)
    ms._connection = None
    init_db()
    assert ms.get_memory("o1").origin == "dehydrate"
    assert ms.list_archived_memories()[0].origin == "dehydrate"
    assert ms.list_pending_links() == []
    init_db()  # 幂等


def test_insert_get_update_link():
    import core.memory_store as ms

    lid = ms.insert_link(
        "n1", "o1", "update", judged_by="rule", status="pending",
        subtype="correction", similarity=0.9, reason="r",
    )
    link = ms.get_link(lid)
    assert (link.src_id, link.dst_id, link.relation) == ("n1", "o1", "update")
    assert (link.subtype, link.similarity, link.reason) == ("correction", 0.9, "r")
    assert link.status == "pending" and link.resolved_at is None

    assert ms.update_link_status(lid, "resolved") is True
    assert ms.get_link(lid).resolved_at
    assert ms.update_link_status(lid, "pending") is True
    assert ms.get_link(lid).resolved_at is None
    assert ms.update_link_status(9999, "resolved") is False
    assert ms.get_link(9999) is None

    rid = ms.insert_link("a", "b", "refine", judged_by="llm", status="resolved")
    assert ms.get_link(rid).resolved_at is not None


def test_list_pending_links_order_and_limit():
    import core.memory_store as ms
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    l1 = ms.insert_link("a", "b", "update", judged_by="rule", status="pending", now=t0)
    l2 = ms.insert_link("c", "d", "update", judged_by="rule", status="pending",
                        now=t0 + timedelta(days=1))
    l3 = ms.insert_link("e", "f", "update", judged_by="rule", status="pending", now=t0)
    ms.insert_link("g", "h", "update", judged_by="rule", status="resolved", now=t0)
    assert [x.id for x in ms.list_pending_links()] == [l2, l3, l1]
    assert [x.id for x in ms.list_pending_links(limit=2)] == [l2, l3]


def test_links_for_and_pending_ids():
    import core.memory_store as ms

    l1 = ms.insert_link("a", "b", "update", judged_by="rule", status="pending")
    l2 = ms.insert_link("c", "a", "refine", judged_by="rule", status="resolved")
    ms.insert_link("x", "y", "update", judged_by="rule", status="resolved")
    assert [x.id for x in ms.links_for(["a"])] == [l1, l2]
    assert [x.id for x in ms.links_for(["b", "zzz"])] == [l1]
    assert ms.links_for([]) == []
    assert ms.pending_link_memory_ids() == {"a", "b"}


def test_delete_memories_cascades_links():
    import core.memory_store as ms

    a, b, c = _mk("甲"), _mk("乙"), _mk("丙")
    ms.insert_link(a, b, "update", judged_by="rule", status="pending")
    ms.insert_link(c, a, "refine", judged_by="rule", status="resolved")
    keep = ms.insert_link(b, c, "update", judged_by="rule", status="pending")
    ms.delete_memories([a])
    assert [x.id for x in ms.links_for([a, b, c])] == [keep]


def test_supersede_and_archive_then_restore_clear():
    import core.memory_store as ms

    old, new = _mk("用户住在北京"), _mk("用户搬到了上海")
    assert ms.supersede_and_archive([old], new) == 1
    assert ms.supersede_and_archive([], new) == 0
    assert ms.get_memory(old) is None
    assert old not in {m.memory_id for m in list_active_memories()}
    arch = ms.list_archived_memories()
    assert [m.memory_id for m in arch] == [old] and arch[0].superseded_by == new

    assert ms.restore_archived([old], clear_superseded=True) == 1
    m = ms.get_memory(old)
    assert m.superseded_by is None
    assert old in {x.memory_id for x in list_active_memories()}


def test_restore_archived_default_keeps_superseded():
    import core.memory_store as ms

    old, new = _mk("甲"), _mk("乙")
    ms.supersede_and_archive([old], new)
    ms.restore_archived([old])
    assert ms.get_memory(old).superseded_by == new


def test_scale_decay_weight_clamps():
    import core.memory_store as ms

    mid = _mk(decay_weight=0.8)
    ms.scale_decay_weight(mid, 0.5)
    assert ms.get_memory(mid).decay_weight == pytest.approx(0.4)
    ms.scale_decay_weight(mid, 10)
    assert ms.get_memory(mid).decay_weight == 1.0
    ms.scale_decay_weight(mid, -1)
    assert ms.get_memory(mid).decay_weight == 0.0


def test_clear_unresolved():
    import core.memory_store as ms

    mid = _mk(unresolved=True)
    assert ms.get_memory(mid).unresolved is True
    ms.clear_unresolved(mid)
    assert ms.get_memory(mid).unresolved is False


def test_memory_stats_link_fields():
    import core.memory_store as ms

    s = ms.memory_stats()
    assert s["pending_conflicts"] == 0 and s["memory_links"] == 0
    ms.insert_link("a", "b", "update", judged_by="rule", status="pending")
    ms.insert_link("c", "d", "refine", judged_by="rule", status="resolved")
    s = ms.memory_stats()
    assert s["pending_conflicts"] == 1 and s["memory_links"] == 2
