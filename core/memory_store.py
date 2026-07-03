"""
Phase 1 MVP — SQLite 记忆存储层。

职责：
- 初始化数据库和表结构。
- 插入 / 查询 / 更新记忆。
- 所有 SQL 使用参数绑定，禁止字符串拼接。
"""

import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import DB_PATH


@dataclass
class Memory:
    """记忆中转数据类。"""

    memory_id: str
    type: str
    content: str
    valence: float = 0.0
    arousal: float = 0.0
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    decay_weight: float = 1.0
    embedding: np.ndarray | None = None
    source_conv_id: str | None = None
    unresolved: bool = False
    tags: list[str] = field(default_factory=list)
    superseded_by: str | None = None


# ── SQL ───────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    content         TEXT NOT NULL,
    valence         REAL DEFAULT 0.0,
    arousal         REAL DEFAULT 0.0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count    INTEGER DEFAULT 0,
    decay_weight    REAL DEFAULT 1.0,
    embedding       BLOB,
    source_conv_id  TEXT,
    unresolved      BOOLEAN DEFAULT FALSE,
    tags            TEXT DEFAULT '',
    superseded_by   TEXT DEFAULT NULL
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_type
    ON memories(type);
"""

INSERT_SQL = """
INSERT INTO memories
    (memory_id, type, content, valence, arousal,
     decay_weight, embedding, source_conv_id,
     unresolved, tags, superseded_by)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

INSERT_WITH_TIME_SQL = """
INSERT INTO memories
    (memory_id, type, content, valence, arousal,
     created_at, last_accessed,
     decay_weight, embedding, source_conv_id,
     unresolved, tags, superseded_by)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

SELECT_ACTIVE_SQL = """
SELECT
    memory_id, type, content, valence, arousal,
    created_at, last_accessed, access_count,
    decay_weight, embedding, source_conv_id,
    unresolved, tags, superseded_by
FROM memories
WHERE superseded_by IS NULL
ORDER BY created_at DESC;
"""

SELECT_RECENT_SQL = """
SELECT
    memory_id, type, content, valence, arousal,
    created_at, last_accessed, access_count,
    decay_weight, embedding, source_conv_id,
    unresolved, tags, superseded_by
FROM memories
WHERE superseded_by IS NULL
ORDER BY created_at DESC
LIMIT ?;
"""

UPDATE_ACCESS_SQL = """
UPDATE memories
SET access_count = access_count + 1,
    last_accessed = CURRENT_TIMESTAMP
WHERE memory_id = ?;
"""

UPDATE_DECAY_SQL = """
UPDATE memories SET decay_weight = ? WHERE memory_id = ?;
"""

MARK_SUPERSEDED_SQL = """
UPDATE memories SET superseded_by = ? WHERE memory_id = ?;
"""

REINFORCE_SQL = """
UPDATE memories
SET decay_weight = ?,
    arousal = ?,
    tags = ?,
    access_count = access_count + 1,
    last_accessed = CURRENT_TIMESTAMP
WHERE memory_id = ?;
"""

GET_META_SQL = """
SELECT value FROM meta WHERE key = ?;
"""

SET_META_SQL = """
INSERT INTO meta (key, value) VALUES (?, ?)
ON CONFLICT(key) DO UPDATE SET value = excluded.value;
"""


# ── 连接管理 ──────────────────────────────────────────────

_connection: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """获取（懒初始化的）数据库连接。"""
    global _connection
    if _connection is None:
        db_dir = Path(DB_PATH).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(str(DB_PATH))
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL;")
        _connection.execute("PRAGMA foreign_keys=ON;")
    return _connection


def close_db() -> None:
    """关闭数据库连接（程序退出时调用）。"""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


# ── 核心 CRUD ─────────────────────────────────────────────


def init_db() -> None:
    """初始化数据库：建表、建索引。幂等（Phase 2 新增 meta 表和 3 条索引）。"""
    conn = _get_conn()
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_INDEX_SQL)
    # Phase 2 新增索引
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_weight);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_unresolved ON memories(unresolved);"
    )
    # Phase 2 meta 表
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "    key   TEXT PRIMARY KEY,"
        "    value TEXT NOT NULL"
        ");"
    )
    conn.commit()


def insert_memory(memory: Memory) -> None:
    """插入单条记忆。当 created_at / last_accessed 非空时显式写入。"""
    conn = _get_conn()
    embedding_blob = None
    if memory.embedding is not None:
        embedding_blob = memory.embedding.astype(np.float32).tobytes()

    has_time = bool(memory.created_at or memory.last_accessed)
    if has_time:
        conn.execute(
            INSERT_WITH_TIME_SQL,
            (
                memory.memory_id,
                memory.type,
                memory.content,
                memory.valence,
                memory.arousal,
                memory.created_at or None,  # None → DB 使用 DEFAULT
                memory.last_accessed or None,
                memory.decay_weight,
                embedding_blob,
                memory.source_conv_id,
                int(memory.unresolved),
                ",".join(memory.tags),
                memory.superseded_by,
            ),
        )
    else:
        conn.execute(
            INSERT_SQL,
            (
                memory.memory_id,
                memory.type,
                memory.content,
                memory.valence,
                memory.arousal,
                memory.decay_weight,
                embedding_blob,
                memory.source_conv_id,
                int(memory.unresolved),
                ",".join(memory.tags),
                memory.superseded_by,
            ),
        )
    conn.commit()


def insert_memories(memories: list[Memory]) -> None:
    """批量插入记忆。单条失败时跳过该条，继续写入其余。"""
    for mem in memories:
        try:
            insert_memory(mem)
        except Exception as e:
            import logging

            logging.warning("写入记忆 %s 失败: %s", mem.memory_id, e)


def list_active_memories() -> list[Memory]:
    """查询所有有效记忆（superseded_by IS NULL）。"""
    conn = _get_conn()
    rows = conn.execute(SELECT_ACTIVE_SQL).fetchall()
    return [_row_to_memory(r) for r in rows]


def list_recent_memories(limit: int = 10) -> list[Memory]:
    """查询最近 N 条有效记忆。"""
    conn = _get_conn()
    rows = conn.execute(SELECT_RECENT_SQL, (limit,)).fetchall()
    return [_row_to_memory(r) for r in rows]


def mark_accessed(memory_ids: list[str]) -> None:
    """批量更新记忆的 access_count 和 last_accessed。"""
    if not memory_ids:
        return
    conn = _get_conn()
    conn.executemany(UPDATE_ACCESS_SQL, [(mid,) for mid in memory_ids])
    conn.commit()


# ── Phase 2：衰减与写入管线存储层 ──────────────────────────


def update_decay_weights(updates: list[tuple[float, str]]) -> None:
    """批量持久化衰减权重。参数为 (new_weight, memory_id) 列表。"""
    if not updates:
        return
    conn = _get_conn()
    conn.executemany(UPDATE_DECAY_SQL, updates)
    conn.commit()


def mark_superseded(old_memory_ids: list[str], new_memory_id: str) -> None:
    """将若干旧记忆标记为被 new_memory_id 覆盖。空列表直接返回。"""
    if not old_memory_ids:
        return
    conn = _get_conn()
    conn.executemany(
        MARK_SUPERSEDED_SQL,
        [(new_memory_id, mid) for mid in old_memory_ids],
    )
    conn.commit()


def reinforce_memory(
    memory_id: str,
    new_decay_weight: float,
    new_arousal: float,
    merged_tags: list[str],
) -> None:
    """emotional 强化更新：提权重、更新唤醒度、合并标签、访问计数 +1。"""
    conn = _get_conn()
    conn.execute(
        REINFORCE_SQL,
        (
            new_decay_weight,
            new_arousal,
            ",".join(merged_tags),
            memory_id,
        ),
    )
    conn.commit()


def get_meta(key: str) -> str | None:
    """读取 meta 值，不存在返回 None。"""
    conn = _get_conn()
    row = conn.execute(GET_META_SQL, (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    """UPSERT 写入 meta（INSERT ... ON CONFLICT DO UPDATE）。"""
    conn = _get_conn()
    conn.execute(SET_META_SQL, (key, value))
    conn.commit()


# ── 内部辅助 ──────────────────────────────────────────────


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """将数据库行转换为 Memory 数据类。"""
    embedding = None
    blob = row["embedding"]
    if blob is not None:
        try:
            embedding = np.frombuffer(blob, dtype=np.float32).copy()
        except Exception:
            embedding = None

    tags_raw = row["tags"] or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    return Memory(
        memory_id=row["memory_id"],
        type=row["type"],
        content=row["content"],
        valence=row["valence"],
        arousal=row["arousal"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        decay_weight=row["decay_weight"],
        embedding=embedding,
        source_conv_id=row["source_conv_id"],
        unresolved=bool(row["unresolved"]),
        tags=tags,
        superseded_by=row["superseded_by"],
    )
