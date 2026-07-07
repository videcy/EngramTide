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


def test_init_db_creates_new_indexes():
    """init_db 后三条新索引存在。"""
    import sqlite3

    from core.memory_store import _get_conn

    conn = _get_conn()
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_memories_%';"
    ).fetchall()
    index_names = {r["name"] for r in indexes}
    assert "idx_memories_type" in index_names
    assert "idx_memories_decay" in index_names
    assert "idx_memories_created" in index_names
    assert "idx_memories_unresolved" in index_names
