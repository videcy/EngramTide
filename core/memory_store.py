"""
Phase 1 MVP — SQLite 记忆存储层。

职责：
- 初始化数据库和表结构。
- 插入 / 查询 / 更新记忆。
- 所有 SQL 使用参数绑定，禁止字符串拼接。
"""

import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

import config
from config import DB_PATH
from utils.time_utils import utc_now

logger = logging.getLogger(__name__)


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

# 唯一的热查询是 `WHERE superseded_by IS NULL ORDER BY created_at DESC`。
# 原先的 idx_memories_type / _decay / _unresolved / _created 对它一条都用不上
# （仍是全表扫 + 排序），却让每次写入都要维护 4 棵 B 树。换成一条部分索引：
# 它同时覆盖过滤条件与排序键，是唯一能真正被用上的那条。
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_active_created
    ON memories(created_at) WHERE superseded_by IS NULL;
"""

# 建过就得删干净——留着只会继续吃写入成本。
DROP_LEGACY_INDEXES_SQL = (
    "DROP INDEX IF EXISTS idx_memories_type;",
    "DROP INDEX IF EXISTS idx_memories_decay;",
    "DROP INDEX IF EXISTS idx_memories_created;",
    "DROP INDEX IF EXISTS idx_memories_unresolved;",
)

# 归档表：列与 memories 完全一致，额外记 archived_at。
# 归档不是删除——数据仍在，只是移出热路径（不进 list_active_memories、不进向量矩阵）。
CREATE_ARCHIVE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories_archive (
    memory_id       TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    content         TEXT NOT NULL,
    valence         REAL DEFAULT 0.0,
    arousal         REAL DEFAULT 0.0,
    created_at      TIMESTAMP,
    last_accessed   TIMESTAMP,
    access_count    INTEGER DEFAULT 0,
    decay_weight    REAL DEFAULT 1.0,
    embedding       BLOB,
    source_conv_id  TEXT,
    unresolved      BOOLEAN DEFAULT FALSE,
    tags            TEXT DEFAULT '',
    superseded_by   TEXT DEFAULT NULL,
    archived_at     TEXT NOT NULL
);
"""

_MEMORY_COLUMNS = (
    "memory_id, type, content, valence, arousal, "
    "created_at, last_accessed, access_count, decay_weight, embedding, "
    "source_conv_id, unresolved, tags, superseded_by"
)

INSERT_SQL = """
INSERT INTO memories
    (memory_id, type, content, valence, arousal,
     access_count, decay_weight, embedding, source_conv_id,
     unresolved, tags, superseded_by)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

INSERT_WITH_TIME_SQL = """
INSERT INTO memories
    (memory_id, type, content, valence, arousal,
     created_at, last_accessed,
     access_count, decay_weight, embedding, source_conv_id,
     unresolved, tags, superseded_by)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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

SELECT_BY_ID_SQL = """
SELECT
    memory_id, type, content, valence, arousal,
    created_at, last_accessed, access_count,
    decay_weight, embedding, source_conv_id,
    unresolved, tags, superseded_by
FROM memories
WHERE memory_id = ?;
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
UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE memory_id = ?;
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


# ── Phase 5：日志表（纯增量埋点，不改生产 schema）──────────

CREATE_DECAY_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS decay_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    ts          TEXT NOT NULL,         -- ISO timestamp
    hours       REAL NOT NULL,        -- Δt since last decay
    n           INTEGER NOT NULL,     -- access_count at event time
    multiplier  REAL NOT NULL,        -- exp(-rate · hours / (24·s))
    floored     INTEGER NOT NULL      -- 1 if clamped to DECAY_FLOOR
);
"""

CREATE_ACCESS_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS access_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    ts          TEXT NOT NULL,
    source      TEXT NOT NULL          -- retrieval / strong_activation / surface / dedup
);
"""


# ── P2：FTS5 关键词倒排索引 ────────────────────────────────

# 外部内容表（content='memories'）——索引不复制一份 content，只存倒排项。
# tokenize='trigram' 是中文的关键：unicode61 会把整句 CJK 切成一个 token，
# 等于索引失效；trigram 按 3 字符滑窗建项，对 CJK 和英文都可用。
CREATE_FTS_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='trigram'
);
"""

CREATE_FTS_TRIGGERS_SQL = (
    """
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES('delete', old.rowid, old.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES('delete', old.rowid, old.content);
        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
    END;
    """,
)

# 本次进程内 FTS5 是否可用。None = 尚未探测。
_fts_available: bool | None = None


def fts_available() -> bool:
    """FTS5 索引本次运行是否可用（建表失败 / 配置关闭 → False）。"""
    return bool(_fts_available)


# ── 连接管理 ──────────────────────────────────────────────

_connection: sqlite3.Connection | None = None

# 记忆集合版本号：只在「集合成员变化」时 +1（插入 / 删除 / supersede / 归档）。
# core.vector_index 用它判断缓存是否过期，避免 memory_store 反向 import 造成循环。
#
# ★ decay_weight 变化**不**碰这个计数器：权重是打分时才乘的标量，不进矩阵。
#   把它算作失效条件会让缓存每轮都重建，收益直接归零。
_data_version: int = 0


def data_version() -> int:
    """当前记忆集合版本号。"""
    return _data_version


def _bump_data_version() -> None:
    """标记记忆集合已变化，使向量缓存失效。"""
    global _data_version
    _data_version += 1


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
    global _connection, _fts_available
    if _connection is not None:
        _connection.close()
        _connection = None
    _fts_available = None
    _bump_data_version()


# ── 核心 CRUD ─────────────────────────────────────────────


def init_db() -> None:
    """初始化数据库：建表、建索引、建归档表与 FTS5 索引。幂等。"""
    global _fts_available

    conn = _get_conn()
    conn.execute(CREATE_TABLE_SQL)
    # Phase 2 meta 表（FTS 回填要读它，先建）
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "    key   TEXT PRIMARY KEY,"
        "    value TEXT NOT NULL"
        ");"
    )
    _migrate_schema(conn)
    for stmt in DROP_LEGACY_INDEXES_SQL:
        conn.execute(stmt)
    conn.execute(CREATE_INDEX_SQL)

    # P4 归档表
    conn.execute(CREATE_ARCHIVE_TABLE_SQL)

    # Phase 5 日志表 + P0 保留期清理所需的 ts 索引
    conn.execute(CREATE_DECAY_EVENTS_SQL)
    conn.execute(CREATE_ACCESS_EVENTS_SQL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decay_events_ts ON decay_events(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_access_events_ts ON access_events(ts);")
    conn.commit()

    _ensure_embedding_dim_meta(conn)

    _fts_available = _init_fts(conn)


def _ensure_embedding_dim_meta(conn: sqlite3.Connection) -> None:
    """
    确保 meta.embedding_dim 有值，供 config.check_embedding_dim() 校验。

    - 空库 → 记当前 EMBEDDING_DIM。
    - 老库没记过 → 从任意一条已存 embedding 的 BLOB 长度反推真实维度并记下。
      如果它与当前配置不一致，启动检查会据此报错并提示重建——这正是我们要的：
      维度不匹配不会抛异常，只会让相似度静默退化成垃圾值，必须显式拦住。
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'embedding_dim'"
    ).fetchone()
    if row is not None:
        return

    blob_row = conn.execute(
        "SELECT length(embedding) AS n FROM memories "
        "WHERE embedding IS NOT NULL LIMIT 1"
    ).fetchone()
    dim = (blob_row["n"] // 4) if blob_row and blob_row["n"] else config.EMBEDDING_DIM

    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('embedding_dim', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(dim),),
    )
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """
    就地补齐旧库缺失的列。幂等。

    superseded_at：记录「何时被取代」，supersede 链的保留期清理需要它。
    旧库没有这一列，回填成当前时间——等于让存量 superseded 记忆从升级这一刻
    开始计 30 天，而不是被当成早已过期立刻硬删。
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories);")}
    if "superseded_at" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN superseded_at TIMESTAMP;")
        conn.execute(
            "UPDATE memories SET superseded_at = ? WHERE superseded_by IS NOT NULL;",
            (utc_now().isoformat(),),
        )


def _init_fts(conn: sqlite3.Connection) -> bool:
    """
    建 FTS5 虚表与同步触发器；首次建成时回填存量数据。

    本机 SQLite 未编译 FTS5 时建表会抛 OperationalError —— 捕获后返回 False，
    检索侧据此降级到 rapidfuzz，不影响任何其他功能。
    """
    if not config.FTS_ENABLED:
        return False
    try:
        existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone() is not None
        conn.execute(CREATE_FTS_TABLE_SQL)
        for stmt in CREATE_FTS_TRIGGERS_SQL:
            conn.execute(stmt)
        if not existed:
            # 触发器只覆盖未来的写入，存量行要显式 rebuild 一次
            conn.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild');"
            )
        conn.commit()
        return True
    except sqlite3.Error as e:
        conn.rollback()
        logger.warning(
            "FTS5 索引不可用（%s），关键词通道降级为 rapidfuzz 全表匹配。", e
        )
        return False


def rebuild_fts_index() -> bool:
    """从 memories 全量重建 FTS5 索引。迁移或触发器缺失后修复用。"""
    if not fts_available():
        return False
    conn = _get_conn()
    try:
        with conn:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
        return True
    except sqlite3.Error as e:
        logger.warning("重建 FTS5 索引失败: %s", e)
        return False


def fts_search(query_text: str, limit: int) -> dict[str, float] | None:
    """
    FTS5 召回关键词候选，返回 {memory_id: 归一化分}，分数 ∈ [0, 1]。

    返回 None 表示**这条查询 FTS5 处理不了**（索引不可用，或查询里没有任何
    ≥3 字符的检索项），调用方应降级到 rapidfuzz；返回 {} 表示查得动但没命中。
    这两种情况必须分开——否则「查不了」会被当成「没有关键词命中」，关键词通道
    就静默失效了。

    bm25() 在 SQLite 里返回的是**越小越相关的负数**，且无上下界；而现有双通道公式
    假设 kw ∈ [0, 1]（原实现是 partial_ratio/100）。因此必须在候选集内做 min-max
    归一化——否则 RETRIEVAL_KEYWORD_WEIGHT 的实际含义会被悄悄改掉。

    单条命中时归一化没有意义（min == max），直接给 1.0。
    """
    if not fts_available():
        return None
    match_expr = _build_fts_match(query_text)
    if not match_expr:
        return None

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT m.memory_id AS memory_id, bm25(memories_fts) AS rank "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match_expr, limit),
        ).fetchall()
    except sqlite3.Error as e:
        logger.warning("FTS5 查询失败 (%s)，本轮降级到模糊匹配。", e)
        return None

    if not rows:
        return {}

    # bm25 越小越相关 → 取负号后越大越相关
    raw = {r["memory_id"]: -float(r["rank"]) for r in rows}
    values = list(raw.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {mid: 1.0 for mid in raw}
    return {mid: (v - lo) / (hi - lo) for mid, v in raw.items()}


# CJK 字符（含中日韩），用于判断一个片段该按「词」还是按「字符 n-gram」处理
_CJK_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# 把自由文本切成「连续 CJK 段」与「拉丁/数字词」，顺带丢掉标点
_SEGMENT = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+"
    r"|[0-9A-Za-z][0-9A-Za-z_'\-.]*"
)

# 单次查询最多带多少个检索项。长句拆出的 3-gram 很多，全带上只会拖慢查询，
# 命中面却没什么增益。
_MAX_FTS_TERMS = 32


def _build_fts_match(query_text: str) -> str:
    """
    把自由文本转成安全的 FTS5 MATCH 表达式；无法构造时返回 ""。

    两件事必须一起处理：

    ① **安全**：用户输入里的 `"`、`*`、`NEAR`、`:`、`(` 都是 FTS5 查询语法字符，
       直接拼进去轻则语法错、重则改变查询语义。每个检索项一律用双引号包成字面量
       短语（内部 `"` 转义成 `""`），再用 OR 连接——命中任一即召回，排序交给 bm25。

    ② **中文**：trigram 分词器要求每个检索项至少 3 个字符，而中文的常用词大多是
       2 个字。所以 CJK 段不能整段当一个词（"我上次在东京吃的拉面" 整段做短语只
       能精确子串匹配），而要拆成滑动 3-gram：「东京吃」「京吃的」…… 这样既能被
       trigram 索引命中，又保留了部分匹配能力。
       长度 < 3 的 CJK 段（如单独查「东京」）无论如何都进不了 trigram 索引，
       此时返回 "" 让调用方降级到模糊匹配。

    拉丁词按原样保留（≥3 字符），这是它们最自然、最精确的形态。
    """
    text = (query_text or "").strip()
    if not text:
        return ""

    terms: list[str] = []
    for seg in _SEGMENT.findall(text):
        if _CJK_CHAR.search(seg):
            if len(seg) >= 3:
                terms.extend(seg[i : i + 3] for i in range(len(seg) - 2))
        elif len(seg) >= 3:
            terms.append(seg)

    # 去重保序，再截断
    seen: set[str] = set()
    unique = [t for t in terms if not (t in seen or seen.add(t))][:_MAX_FTS_TERMS]
    if not unique:
        return ""

    return " OR ".join('"' + t.replace('"', '""') + '"' for t in unique)


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
                memory.access_count,
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
                memory.access_count,
                memory.decay_weight,
                embedding_blob,
                memory.source_conv_id,
                int(memory.unresolved),
                ",".join(memory.tags),
                memory.superseded_by,
            ),
        )
    conn.commit()
    _bump_data_version()


def _insert_params(memory: Memory, with_time: bool) -> tuple:
    """把 Memory 摊平成 INSERT 参数元组。with_time 决定用哪条 SQL 的列序。"""
    embedding_blob = None
    if memory.embedding is not None:
        embedding_blob = memory.embedding.astype(np.float32).tobytes()
    head = (
        memory.memory_id,
        memory.type,
        memory.content,
        memory.valence,
        memory.arousal,
    )
    times = (memory.created_at or None, memory.last_accessed or None) if with_time else ()
    tail = (
        memory.access_count,
        memory.decay_weight,
        embedding_blob,
        memory.source_conv_id,
        int(memory.unresolved),
        ",".join(memory.tags),
        memory.superseded_by,
    )
    return head + times + tail


def insert_memories(memories: list[Memory]) -> None:
    """
    批量插入记忆：单事务 executemany，整批一次 fsync。

    会话结束一次写 8~64 条，逐条 commit 就是 64 次 fsync。

    按「是否自带时间戳」分成两批走不同的 SQL——带时间的走显式列，不带的走省略
    created_at/last_accessed 的那条，让 DB 的 DEFAULT CURRENT_TIMESTAMP 生效。
    （把 None 显式塞进列里会写成 NULL，DEFAULT 不会触发。）

    保留「单条失败不牵连其余」的语义：批量提交失败时整批回退为逐条写入，
    只有真正有问题的那条被跳过并记 warning。
    """
    if not memories:
        return

    with_time = [m for m in memories if m.created_at or m.last_accessed]
    without_time = [m for m in memories if not (m.created_at or m.last_accessed)]

    conn = _get_conn()
    try:
        with conn:
            if with_time:
                conn.executemany(
                    INSERT_WITH_TIME_SQL,
                    [_insert_params(m, True) for m in with_time],
                )
            if without_time:
                conn.executemany(
                    INSERT_SQL,
                    [_insert_params(m, False) for m in without_time],
                )
        _bump_data_version()
        return
    except sqlite3.Error as e:
        logger.warning("批量写入 %d 条记忆失败 (%s)，回退逐条写入。", len(memories), e)

    for mem in memories:
        try:
            insert_memory(mem)
        except Exception as e:
            logger.warning("写入记忆 %s 失败: %s", mem.memory_id, e)


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


def get_memory(memory_id: str) -> Memory | None:
    """按 ID 查询单条记忆；包括已经被 supersede 的记录。"""
    conn = _get_conn()
    row = conn.execute(SELECT_BY_ID_SQL, (memory_id,)).fetchone()
    return _row_to_memory(row) if row else None


def delete_memories(memory_ids: list[str]) -> int:
    """彻底删除记忆及其衰减/访问日志，返回实际删除的记忆数量。"""
    unique_ids = list(dict.fromkeys(mid for mid in memory_ids if mid))
    if not unique_ids:
        return 0

    conn = _get_conn()
    placeholders = ",".join("?" for _ in unique_ids)
    with conn:
        conn.execute(
            f"DELETE FROM decay_events WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
        conn.execute(
            f"DELETE FROM access_events WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
        cursor = conn.execute(
            f"DELETE FROM memories WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
    _bump_data_version()
    return cursor.rowcount


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
    ts = utc_now().isoformat()
    conn = _get_conn()
    conn.executemany(
        MARK_SUPERSEDED_SQL,
        [(new_memory_id, ts, mid) for mid in old_memory_ids],
    )
    conn.commit()
    _bump_data_version()


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


# ── Phase 5 日志函数（纯增量埋点）─────────────────────────


def log_decay_event(
    memory_id: str,
    ts: str,
    hours: float,
    n: int,
    multiplier: float,
    floored: bool,
) -> None:
    """记录一次衰减事件。"""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO decay_events (memory_id, ts, hours, n, multiplier, floored) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (memory_id, ts, hours, n, multiplier, int(floored)),
    )
    conn.commit()


def log_access_event(
    memory_id: str,
    source: str,
) -> None:
    """记录一次计访问事件，带来源标签。source ∈ {retrieval, strong_activation, surface, dedup}。"""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO access_events (memory_id, ts, source) VALUES (?, ?, ?)",
        (memory_id, ts, source),
    )
    conn.commit()


def log_decay_events_batch(rows: list[tuple]) -> None:
    """
    批量落库衰减事件，单事务一次 fsync。

    rows 每项为 (memory_id, ts, hours, n, multiplier, floored)。
    逐行 commit 时 N=5000 就是 5000 次 fsync，会话启动能被拖到几十秒——
    这是改成批量的全部理由。
    """
    if not rows:
        return
    conn = _get_conn()
    with conn:
        conn.executemany(
            "INSERT INTO decay_events (memory_id, ts, hours, n, multiplier, floored) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def log_access_events_batch(rows: list[tuple[str, str]]) -> None:
    """批量落库访问事件，单事务。rows 每项为 (memory_id, source)，ts 由本函数统一取。"""
    if not rows:
        return
    ts = utc_now().isoformat()
    conn = _get_conn()
    with conn:
        conn.executemany(
            "INSERT INTO access_events (memory_id, ts, source) VALUES (?, ?, ?)",
            [(mid, ts, source) for mid, source in rows],
        )


def prune_event_logs(retention_days: int | None = None) -> int:
    """
    删除超过保留期的埋点事件，返回删除行数。

    注意 SQLite 删除只把页面标记为空闲，磁盘不会自动缩小——真要回收空间得跑
    vacuum_db()。日常调用这个控制行数增长即可。
    """
    days = (
        config.EVENT_LOG_RETENTION_DAYS if retention_days is None else retention_days
    )
    if days <= 0:
        return 0
    cutoff = (utc_now() - timedelta(days=days)).isoformat()
    conn = _get_conn()
    with conn:
        c1 = conn.execute("DELETE FROM decay_events WHERE ts < ?", (cutoff,))
        deleted = c1.rowcount
        c2 = conn.execute("DELETE FROM access_events WHERE ts < ?", (cutoff,))
        deleted += c2.rowcount
    return deleted


def vacuum_db() -> None:
    """VACUUM 回收删除后释放的页面。耗时随库体积增长，不要放在热路径。"""
    conn = _get_conn()
    conn.isolation_level = None
    try:
        conn.execute("VACUUM;")
    finally:
        conn.isolation_level = ""


def query_decay_events(memory_id: str) -> list[dict]:
    """查询某条记忆的全部衰减事件（供分析用）。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT ts, hours, n, multiplier, floored FROM decay_events "
        "WHERE memory_id = ? ORDER BY ts",
        (memory_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def query_access_events(memory_id: str) -> list[dict]:
    """查询某条记忆的全部访问事件。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT ts, source FROM access_events WHERE memory_id = ? ORDER BY ts",
        (memory_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── P4：冷归档与增长治理 ──────────────────────────────────


def collect_archive_candidates(now=None) -> list[str]:
    """
    挑出可以移出热路径的记忆 id。

    四个条件同时满足才入选（保守到只动最安全的那一类）：
      type == 'episodic'                 只归档情景记忆
      decay_weight <= DECAY_FLOOR + eps  已经完全沉底
      access_count == 0                  从未被检索/激活/浮现用过
      last_accessed 早于 ARCHIVE_IDLE_DAYS
      superseded_by IS NULL              supersede 链走 purge_superseded()

    access_count > 0 的记忆永远不会被归档——真正被用过的东西留在热表，
    这是刻意的偏置。
    """
    from config import ARCHIVE_IDLE_DAYS, DECAY_FLOOR

    now = now or utc_now()
    cutoff = (now - timedelta(days=ARCHIVE_IDLE_DAYS)).isoformat()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT memory_id FROM memories "
        "WHERE type = 'episodic' "
        "  AND superseded_by IS NULL "
        "  AND access_count = 0 "
        "  AND decay_weight <= ? "
        "  AND last_accessed IS NOT NULL "
        "  AND last_accessed < ?",
        (DECAY_FLOOR + 1e-9, cutoff),
    ).fetchall()
    return [r["memory_id"] for r in rows]


def archive_memories(memory_ids: list[str], now=None) -> int:
    """
    把记忆整行搬到 memories_archive，然后从热表删除。返回搬走的条数。

    归档不是删除：内容与 embedding 全部保留，search_archive() 还能翻出来。
    对休眠唤醒机制的影响仅限于「不再自动参与逐轮激活」。
    """
    unique_ids = list(dict.fromkeys(mid for mid in memory_ids if mid))
    if not unique_ids:
        return 0

    archived_at = (now or utc_now()).isoformat()
    conn = _get_conn()
    placeholders = ",".join("?" for _ in unique_ids)
    with conn:
        cursor = conn.execute(
            f"INSERT OR REPLACE INTO memories_archive ({_MEMORY_COLUMNS}, archived_at) "
            f"SELECT {_MEMORY_COLUMNS}, ? FROM memories "
            f"WHERE memory_id IN ({placeholders})",
            [archived_at, *unique_ids],
        )
        moved = cursor.rowcount
        conn.execute(
            f"DELETE FROM decay_events WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
        conn.execute(
            f"DELETE FROM access_events WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
        conn.execute(
            f"DELETE FROM memories WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
    _bump_data_version()
    return moved


def purge_superseded(retention_days: int | None = None, now=None) -> int:
    """
    硬删除被标记 supersede 超过保留期的旧记忆，返回删除条数。

    这类记忆已被新记忆完整取代，留着的唯一价值是审计追溯，默认 30 天足够——
    所以走硬删而不是归档。
    """
    from config import SUPERSEDED_RETENTION_DAYS

    days = SUPERSEDED_RETENTION_DAYS if retention_days is None else retention_days
    if days < 0:
        return 0
    cutoff = ((now or utc_now()) - timedelta(days=days)).isoformat()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT memory_id FROM memories "
        "WHERE superseded_by IS NOT NULL AND superseded_at IS NOT NULL "
        "  AND superseded_at < ?",
        (cutoff,),
    ).fetchall()
    ids = [r["memory_id"] for r in rows]
    if not ids:
        return 0
    return delete_memories(ids)


def list_archived_memories(limit: int | None = None) -> list[Memory]:
    """读取归档表（供 search_archive 与 /stats 使用）。"""
    conn = _get_conn()
    sql = (
        f"SELECT {_MEMORY_COLUMNS} FROM memories_archive "
        f"ORDER BY archived_at DESC"
    )
    if limit is not None:
        rows = conn.execute(sql + " LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    return [_row_to_memory(r) for r in rows]


def restore_archived(memory_ids: list[str]) -> int:
    """把归档记忆搬回热表（显式召回用）。返回恢复条数。"""
    unique_ids = list(dict.fromkeys(mid for mid in memory_ids if mid))
    if not unique_ids:
        return 0
    conn = _get_conn()
    placeholders = ",".join("?" for _ in unique_ids)
    with conn:
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO memories ({_MEMORY_COLUMNS}) "
            f"SELECT {_MEMORY_COLUMNS} FROM memories_archive "
            f"WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
        restored = cursor.rowcount
        conn.execute(
            f"DELETE FROM memories_archive WHERE memory_id IN ({placeholders})",
            unique_ids,
        )
    _bump_data_version()
    return restored


def count_active_by_type() -> dict[str, int]:
    """活跃记忆的按类型计数。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT type, COUNT(*) AS n FROM memories "
        "WHERE superseded_by IS NULL GROUP BY type"
    ).fetchall()
    return {r["type"]: r["n"] for r in rows}


def memory_stats() -> dict:
    """
    增长治理的可观测面：活跃 / 归档 / supersede 计数 + 近 30 天净增长 + DB 体积。

    净增长率是判断治理是否生效的唯一指标——只看总量看不出来。
    """
    conn = _get_conn()
    month_ago = (utc_now() - timedelta(days=30)).isoformat()

    by_type = count_active_by_type()
    archived = conn.execute("SELECT COUNT(*) FROM memories_archive").fetchone()[0]
    superseded = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
    ).fetchone()[0]
    new_this_month = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_at >= ?", (month_ago,)
    ).fetchone()[0]
    archived_this_month = conn.execute(
        "SELECT COUNT(*) FROM memories_archive WHERE archived_at >= ?", (month_ago,)
    ).fetchone()[0]
    decay_events = conn.execute("SELECT COUNT(*) FROM decay_events").fetchone()[0]
    access_events = conn.execute("SELECT COUNT(*) FROM access_events").fetchone()[0]

    # WAL 模式下新写入的页先落在 -wal 文件里，只看主库文件会严重低估体积
    db_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            db_bytes += Path(str(DB_PATH) + suffix).stat().st_size
        except OSError:
            pass

    return {
        "active_total": sum(by_type.values()),
        "active_by_type": by_type,
        "archived": archived,
        "superseded": superseded,
        "new_last_30d": new_this_month,
        "archived_last_30d": archived_this_month,
        "net_growth_last_30d": new_this_month - archived_this_month,
        "decay_events": decay_events,
        "access_events": access_events,
        "db_bytes": db_bytes,
        "fts_enabled": fts_available(),
    }
