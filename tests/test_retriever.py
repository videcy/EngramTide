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
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=embedding,
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
