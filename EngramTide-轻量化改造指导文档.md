# EngramTide 轻量化与记忆增长治理 — 开发指导文档

> 版本：v1 · 基于 master 分支（4 commits）代码审阅
> 目标：在不改变现有语义（衰减/激活/浮现/写入策略）的前提下，消除性能热点、补上真正缺失的索引、并给记忆规模加上界。

---

## 0. 总览

### 0.1 现状诊断（按紧急度排序）

| 优先级 | 问题 | 影响面 | 触发规模 |
|---|---|---|---|
| **P0** | 埋点日志逐行 `commit()`，且无 retention | 会话启动阻塞、DB 体积失控 | N > 1000 即明显 |
| **P1** | 每轮两次全表加载 + Python 逐条余弦 | 每轮延迟随 N 线性增长 | N > 2000 明显 |
| **P2** | 关键词通道无索引（`fuzz.partial_ratio` 全表） | 比向量通道更慢 | N > 2000 明显 |
| **P3** | embedding 全维 float32 存储 | DB 体积、内存、矩阵乘耗时 ×3 | 全程 |
| **P4** | 记忆只增不减，无归档/删除路径 | 长期必然爆库 | 累积 |

### 0.2 关键判断：**先不要上 ANN 向量索引**

「没做索引」的直觉修法是接 faiss / HNSW，但这与「轻量化」目标相反：

- N < 2~5 万时，**L2 归一化 + numpy 矩阵乘的暴力检索比 HNSW 更快**，且不需要额外索引内存与重建流程
- HNSW 引入的是：新依赖 + 索引持久化 + 增删同步 + 参数调优，全是「变重」
- 真正缺索引的是**关键词通道**（见 P2），那里可以用 SQLite 内置的 FTS5，零新依赖

**结论：** 先把暴力检索做对（P1 + P3），把倒排索引补上（P2），预留 `search_backend` 抽象层，等 N 真的越过 2 万再切 `sqlite-vec`。

### 0.3 建议实施顺序

```
P0 日志开关          ≈ 0.5 天  ← 立刻做，独立可上线
P1 共用加载 + 矩阵化  ≈ 1 天
P4-② 冷归档          ≈ 1 天    ← 先建出口，再收源头
P4-① episodic 去重   ≈ 0.5 天
P3 embedding 降维    ≈ 0.5 天 + 一次全库重建
P2 FTS5 关键词索引    ≈ 1.5 天
```

---

## P0 — 埋点日志：立刻关闭默认写入

### 现状

`core/memory_store.py`：

```python
def log_decay_event(...):
    conn.execute("INSERT INTO decay_events ...", (...))
    conn.commit()          # ← 每行一次 commit = 每行一次 fsync

def log_access_event(...):
    conn.execute("INSERT INTO access_events ...", (...))
    conn.commit()          # ← 同上
```

`core/decay.py:166`，`apply_decay()` 循环内：

```python
for mem in memories:
    ...
    _log_decay_event(mem.memory_id, hours_elapsed, ...)   # 每条记忆一次
```

### 问题

1. **会话启动阻塞**：`run_decay_update()` 在每次 `start_session()` 时对全部可衰减记忆跑一遍。N = 5000 → 5000 次独立事务提交。WAL 模式下 fsync 仍然是毫秒级，合计可达数十秒。
2. **DB 体积反超主表**：`decay_events` 每次会话每条记忆写一行。500 次会话 × 5000 条 = 250 万行，远超 memories 表本身。
3. **定位错配**：这两张表是 Phase 5 的论文消融埋点（`ABL_ACCESS_*` 系列的观测面），生产运行时不需要。

### 改法

**① 加环境开关，生产默认关**

`config.py`：

```python
# ── Phase 5 埋点（论文实验用，生产默认关闭）─────────────
DECAY_LOG_ENABLED: bool = (
    _get_env("DECAY_LOG_ENABLED", "false").lower() == "true"
)
ACCESS_LOG_ENABLED: bool = (
    _get_env("ACCESS_LOG_ENABLED", "false").lower() == "true"
)
EVENT_LOG_RETENTION_DAYS: int = int(_get_env("EVENT_LOG_RETENTION_DAYS", "30"))
```

> 注意：`core/decay.py:41` 目前用 `__import__("os").getenv(...)` 在模块顶层读取，
> 与 `config.py` 的读取方式不一致。统一收敛到 `config.py`，避免两处默认值漂移。

**② 批量提交**

`apply_decay()` 内改为收集事件，循环结束后一次性落库：

```python
# apply_decay 内
decay_events: list[tuple] = []
for mem in memories:
    ...
    if DECAY_LOG_ENABLED:
        decay_events.append((mem.memory_id, ts, hours_elapsed, mem.access_count,
                             multiplier, int(new_weight <= DECAY_FLOOR)))
...
if decay_events:
    log_decay_events_batch(decay_events)     # executemany + 单次 commit
```

`memory_store.py` 新增：

```python
def log_decay_events_batch(rows: list[tuple]) -> None:
    if not rows:
        return
    conn = _get_conn()
    with conn:                                # 单事务
        conn.executemany(
            "INSERT INTO decay_events (memory_id, ts, hours, n, multiplier, floored) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows
        )
```

同样为 `log_access_event` 提供 `log_access_events_batch(list[tuple[str, str]])`。

**③ retention 清理**

在 `init_db()` 或 `run_decay_update()` 结束时执行一次：

```python
def prune_event_logs(retention_days: int) -> int:
    cutoff = (utc_now() - timedelta(days=retention_days)).isoformat()
    conn = _get_conn()
    with conn:
        c1 = conn.execute("DELETE FROM decay_events WHERE ts < ?", (cutoff,))
        c2 = conn.execute("DELETE FROM access_events WHERE ts < ?", (cutoff,))
    return c1.rowcount + c2.rowcount
```

配合 `ts` 上的索引：

```sql
CREATE INDEX IF NOT EXISTS idx_decay_events_ts  ON decay_events(ts);
CREATE INDEX IF NOT EXISTS idx_access_events_ts ON access_events(ts);
```

> 定期 `VACUUM` 才能真正回收磁盘空间（SQLite 删除后页面只标记为空闲）。可加一个
> `/maintenance` 命令，或在 `close_db()` 前按频率触发。

### 验证

- 现有测试中断言 `query_decay_events()` / `query_access_events()` 的用例，需要在 fixture 里显式打开开关。这是本次改动最容易踩的回归点。
- 手工验证：N=3000 的库，`start_session()` 耗时改造前后对比。

### 风险

低。埋点关闭不影响任何记忆语义，`ABL_ACCESS_*` 消融开关（控制**是否计访问**）与日志开关（控制**是否记录**）是正交的两件事，不要混。

---

## P1 — 每轮两次全表扫描 + 逐条余弦

### 现状

`main.py`：

```python
activation_report = context_aware_update(query_embedding, ...)   # :302 → memories=None
retrieval_details = retrieve_memories_detailed(query_embedding, ...)  # :328 → memories=None
```

`engramtide/api.py:280 / :291` 同样两处都不传 `memories`。

两个函数内部各自调用一次 `list_active_memories()`，即：

- 每轮 **2 次**全表 `SELECT` + 2 次全部 embedding BLOB 的 `np.frombuffer` 反序列化
- 每轮 **2 遍** Python for 循环调 `cosine_similarity()`

1536 维 float32 = 6 KB/条。N = 10000 → 每轮从磁盘搬运 120 MB、做 2 万次 Python 层函数调用。

### 改法

**① 共用一次加载（改动最小，先做）**

```python
all_mems = list_active_memories()

activation_report = context_aware_update(
    query_embedding, memories=all_mems, exclude_mild_ids=...
)
retrieval_details = retrieve_memories_detailed(
    query_embedding, memories=all_mems, top_k=TOP_K_RETRIEVE, query_text=user_input,
)
```

> **注意顺序依赖**：`context_aware_update()` 会写回 `decay_weight`，而检索打分要乘
> `decay_weight`。目前是「激活落库 → 检索重新读库」，共用列表后检索读到的是**旧权重**。
> 必须让 `compute_activations()` 返回的新权重同步回内存对象，或让
> `context_aware_update` 返回更新后的 `memories`。**这是本项改动的唯一正确性风险点，
> 务必写一条对照测试**（同一输入，改造前后 Top-K 的 memory_id 序列应完全一致）。

**② 向量化（收益主体）**

思路：cosine(a, b) = â · b̂。只要**入库时就把 embedding L2 归一化**，检索时余弦相似度退化为一次矩阵乘。

`core/embedding.py`，`embed_text()` 返回前：

```python
vec = np.asarray(raw, dtype=np.float32)
norm = np.linalg.norm(vec)
return vec / norm if norm > 0 else vec
```

新建 `core/vector_index.py`（进程内常驻缓存）：

```python
class VectorCache:
    """active memories 的 (N, D) 归一化矩阵，与 memory_id 顺序一一对应。"""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None   # (N, D) float32, 已归一化
        self._dirty = True

    def rebuild(self, memories: list[Memory]) -> None:
        rows = [(m.memory_id, m.embedding) for m in memories if m.embedding is not None]
        self._ids = [r[0] for r in rows]
        self._matrix = np.vstack([r[1] for r in rows]) if rows else None
        self._dirty = False

    def similarities(self, q: np.ndarray) -> tuple[list[str], np.ndarray]:
        if self._matrix is None:
            return [], np.array([], dtype=np.float32)
        return self._ids, self._matrix @ q      # 一次矩阵乘，得到全部 sim
```

调用侧：算出 `sims` 数组后，激活与检索**共用同一个数组**，各自套自己的阈值/过滤逻辑。
每轮的相似度计算从 2N 次 Python 调用降到 1 次 BLAS 矩阵乘。

**③ 缓存失效策略（保持简单）**

只在这三个时机置 `_dirty = True`：`insert_memory` / `mark_superseded` / `delete_memories`。
`decay_weight` 变化**不影响矩阵**（权重是打分时才乘上去的标量，不进矩阵），这一点让缓存失效逻辑非常干净——不要把 decay_weight 预乘进矩阵。

### 预期收益

N = 10000、D = 1536：每轮相似度计算从 ~400 ms 降到 ~5 ms；磁盘 IO 从 120 MB/轮 降到 0。

### 顺带清理：无效索引

`init_db()` 建的四个索引，对唯一的热查询
`WHERE superseded_by IS NULL ORDER BY created_at DESC` **一个都用不上**（仍是全表扫 + 排序），却在每次写入时增加 B 树维护成本。

```sql
-- 删掉 idx_memories_type / idx_memories_decay / idx_memories_unresolved
-- idx_memories_created 换成部分索引：
CREATE INDEX IF NOT EXISTS idx_active_created
    ON memories(created_at) WHERE superseded_by IS NULL;
```

### 顺带清理：批量插入

`insert_memories()` 是 `for` 循环调 `insert_memory()`，每条一次 `commit()`。
会话结束时一次写入 8~64 条 → 64 次 fsync。改成单事务 `executemany`，
但保留「单条失败不牵连其余」的语义（先校验，再批量提交；失败条目单独重试并记录）。

---

## P2 — 关键词通道：用 FTS5 替换全表模糊匹配

### 现状

`core/retriever.py`，对**每一条**通过过滤的记忆：

```python
kw = fuzz.partial_ratio(query_text, mem.content) / 100.0
```

`partial_ratio` 是滑动窗口的编辑距离，复杂度约 O(len(query) × len(content))。N 条记忆就是 N 次。
即使 rapidfuzz 底层是 C++，这仍然是**全表扫描且无任何索引**——实测通常比向量通道更慢。

### 改法

SQLite **FTS5** 是内置模块（`sqlite3` 标准库直接可用），零新依赖。

**① 建外部内容表（避免 content 存两份）**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='trigram'          -- 中文关键：trigram 对 CJK 有效，unicode61 会整句成一个 token
);
```

用触发器保持同步：

```sql
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
```

**② 检索改为「候选召回 + 融合」两阶段**

```python
# 阶段 1：FTS5 召回关键词候选（倒排索引，O(log N)）
kw_hits: dict[str, float] = fts_search(query_text, limit=50)   # memory_id -> bm25 归一化分

# 阶段 2：向量分（来自 P1 的矩阵乘，已是全量）
# 融合：kw 分只对命中的候选生效，未命中记 0
score = (RETRIEVAL_VECTOR_WEIGHT * sim + RETRIEVAL_KEYWORD_WEIGHT * kw) * decay_weight
```

**③ 分数归一化**

BM25 是无上界的，而现有公式假设 `kw ∈ [0, 1]`（`partial_ratio / 100`）。
必须把候选集内的 bm25 做 min-max 归一化到 [0, 1]，否则 `RETRIEVAL_KEYWORD_WEIGHT=0.3`
这个权重的含义会变，专有名词兜底会过强或过弱。

**④ 保留降级路径**

`KEYWORD_CHANNEL_ENABLED=false` 或 FTS5 建表失败 → 退化纯向量，与现有行为一致。
rapidfuzz 可作为二级 fallback 保留，也可以直接移除依赖。

### 验证

准备一组「专有名词命中但语义相似度低」的样本（关键词通道存在的意义就在这里），
对比 FTS5 版本和 rapidfuzz 版本的 Top-K 召回，确认没有丢失兜底能力。

### 什么时候才考虑向量 ANN

设一个明确的切换阈值，写进 config：

```python
ANN_BACKEND_THRESHOLD: int = 20000   # active memories 超过此数，提示切换 sqlite-vec
```

到达前保持暴力矩阵乘。切换时优先 `sqlite-vec`（单个扩展文件，无独立服务、无额外进程），
而非 faiss / Milvus。

---

## P3 — embedding 降维：改一个参数换 3 倍

### 原理

OpenAI `text-embedding-3-*` 系列使用 **Matryoshka Representation Learning**，
前 k 维本身就是一个有效的低维表示，直接截断的检索质量损失很小
（官方数据：3-small 从 1536 → 512 维，MTEB 检索指标下降约 1~2 个点）。

### 改法

`core/embedding.py` 请求体加参数：

```python
payload = {
    "model": EMBEDDING_MODEL,
    "input": text,
    "dimensions": EMBEDDING_DIM,      # config: 默认 512
}
```

> 兼容性：`dimensions` 是 OpenAI 规范的可选字段，部分自建 provider（如 Qwen3-Embedding
> 的某些部署）不支持。做一次探测：首次请求带 `dimensions`，返回 400 则回退不带参数，
> 并在本地做截断 + 重新归一化（MRL 模型截断后必须重新 L2 归一化）。

### 迁移

维度变了，**旧 embedding 与新 embedding 不可混用**。需要一次性全库重建：

```
scripts/rebuild_embeddings.py
  1. 读取全部 memories（含 superseded，归档表也要）
  2. 分批调用 embedding API（注意限流与断点续传）
  3. 写回 BLOB，同时做 L2 归一化（配合 P1）
  4. 在 meta 表记录 embedding_model + embedding_dim + rebuilt_at
```

在 `check_config()` 里加一条校验：meta 记录的维度与当前 `EMBEDDING_DIM` 不一致 → 启动时报错并提示重建。**这个校验很重要**，否则维度不匹配会以「相似度全是垃圾值」的形式静默失败。

### 收益

- BLOB 6 KB → 2 KB，DB 体积与内存矩阵均降至 1/3
- 矩阵乘耗时降至 1/3
- 阈值需要**重新校准**：`SIMILARITY_MID` / `SIMILARITY_HIGH` 是在特定 embedding 空间下标定的（源码注释记录了 MiniLM 的 P95_unrelated / P25_related 协议）。降维后必须按同一协议重跑校准，不要沿用旧值。

---

## P4 — 记忆增长治理：补上「出口」

### 核心诊断

当前系统**没有任何删除路径**：

- `DECAY_FLOOR = 0.01` → 记忆权重永不归零，只是沉底到检索不到
- `superseded_by` 标记的旧记忆 → 永久占据表空间与向量矩阵（虽然被检索过滤，但仍参与 `list_active_memories` 的读取与反序列化…… 实际上 `SELECT_ACTIVE_SQL` 已过滤，但 `delete_memories` 是唯一真正删除的入口，且只有用户显式调用）
- `/consolidate` 是**手动**命令；`CONSOLIDATE_SUGGEST_COUNT = 500` 只打印一句提示

**写入速率**：单次会话上限 = `MAX_TOPIC_SEGMENTS(8) × DEHYDRATE_MAX_ITEMS(8)` = 64 条。
每天 1 次会话 → 一年 2 万条。这是纯增量，没有对冲。

### ① 源头：给 episodic 加新颖度门槛

`core/memory_writer.py:92`：

```python
else:
    # episodic 或未知类型 → 直接插入
    insert_memory(new_mem)
```

四类记忆中，semantic 有 0.85 覆盖检测、emotional 有 0.85 强化检测、procedural 有 0.90 去重，
**只有 episodic 无条件 insert**——而它恰恰是数量最大、衰减最快的类型。

改法（复用 emotional 的强化思路，而非 semantic 的覆盖思路）：

```python
EPISODIC_DEDUP_THRESHOLD: float = 0.88     # 建议略高于 semantic 的 0.85

def _handle_episodic(new_mem, existing_episodic, report):
    """近重复 → 强化已有记忆（提权重 + 计访问），不新增条目。"""
    best, best_sim = _find_most_similar(new_mem, existing_episodic)
    if best is not None and best_sim >= EPISODIC_DEDUP_THRESHOLD:
        reinforce_memory(best.memory_id,
                         min(1.0, best.decay_weight + EPISODIC_REINFORCE_BOOST),
                         best.arousal, _merge_tags(best.tags, new_mem.tags))
        report.reinforced += 1
        return
    insert_memory(new_mem)
    report.inserted += 1
```

> **语义权衡**：episodic 是「情景记忆」，理论上两次相似但不同时刻的事件应当各自成条。
> 阈值定高（0.88+）意味着只拦截「同一件事被反复脱水出来」的真重复。
> 上线前先用 `preview` 模式跑一遍历史库，看看会拦掉哪些——如果拦掉了明显不该合的，调高阈值。

### ② 出口：冷归档（本项最关键）

新建归档表，结构与 `memories` 一致：

```sql
CREATE TABLE IF NOT EXISTS memories_archive (
    -- 与 memories 完全相同的列
    ...,
    archived_at TEXT NOT NULL
);
```

归档条件（保守起见，只动最安全的那一类）：

```python
def collect_archive_candidates(now) -> list[str]:
    """
    同时满足：
      type == 'episodic'                       # 只归档情景记忆
      AND decay_weight <= DECAY_FLOOR + 1e-9   # 已完全沉底
      AND access_count == 0                    # 从未被检索/激活/浮现使用过
      AND last_accessed 早于 ARCHIVE_IDLE_DAYS (默认 90 天)
      AND superseded_by IS NULL                # supersede 链另行处理
    """
```

同时处理 supersede 链：

```python
# superseded_by 非空 且 被标记超过 SUPERSEDED_RETENTION_DAYS (默认 30 天) → 直接硬删
# 理由：它已被新记忆完整取代，保留的唯一价值是审计追溯，30 天足够
```

执行时机：`start_session()` 的衰减更新之后，按频率触发（如每 20 次会话一次，
用 meta 表记 `last_archive_run_at`），避免每次启动都扫。

配套：
- 从 `VectorCache` 中移除已归档 id（触发 rebuild）
- 提供 `search_archive(query)` 显式接口，让用户在需要时能翻出来
- **归档不是删除**：数据仍在，只是不进热路径。这比硬删除安全得多，也更容易接受

**效果**：热表规模从「无上界线性增长」变成「由 90 天窗口 + 访问行为决定的稳态」。
被真正用过的记忆（access_count > 0）永远不会被归档,这是正确的偏置。

### ③ 频率：consolidate 自动化

```python
# session.end() 内，写入完成后：
if active_episodic_emotional_count() > CONSOLIDATE_AUTO_THRESHOLD:   # 如 300
    pairs = find_consolidation_candidates(preview=True)
    if len(pairs) >= CONSOLIDATE_MIN_PAIRS:                          # 如 5
        await consolidate(max_pairs=CONSOLIDATE_MAX_PAIRS)
```

注意 `find_consolidation_candidates` 目前是 **O(N²)** 的两两比较。
接入 P1 的矩阵后可以直接算 `M @ M.T` 的上三角，但 N=10000 时这是 10⁸ 个 float（400 MB）。
需要分块计算（每次取 1000 行 × 全量），或者只在同类型内比较（现有逻辑已经是同类型）。
**这是自动化之前必须先解决的性能前提。**

### ④ 可观测性

加一个 `/stats` 命令，让增长趋势可见：

```
活跃记忆    1,247  (semantic 312 / episodic 689 / emotional 201 / procedural 45)
已归档        356
已 supersede   89
本月新增      142   (insert 118 / reinforce 24)
本月归档       67
DB 体积     18.4 MB  (memories 12.1 / fts 3.2 / events 3.1)
```

净增长率是判断治理是否生效的唯一指标。

---

## 附录 A：改动影响面速查

| 改动 | 触碰文件 | 现有测试影响 |
|---|---|---|
| P0 日志开关 | `config.py`, `memory_store.py`, `decay.py` | 埋点相关用例需显式开开关 |
| P1 共用加载 | `main.py`, `engramtide/api.py` | **需新增权重同步对照测试** |
| P1 矩阵化 | 新增 `core/vector_index.py`, `retriever.py`, `decay.py`, `embedding.py` | 相似度精度可能有 float32 累加差异，断言用 `pytest.approx` |
| P1 索引清理 | `memory_store.py` | 无 |
| P2 FTS5 | `memory_store.py`, `retriever.py` | 双通道打分用例需重写 kw 分来源 |
| P3 降维 | `config.py`, `embedding.py`, 新增迁移脚本 | **阈值校准用例全部失效，需重跑校准** |
| P4 episodic 去重 | `memory_writer.py` | 写入管线用例新增分支 |
| P4 归档 | `memory_store.py`, `decay.py`, 新增 schema | 新增用例 |

## 附录 B：不建议做的事

- **不要上 faiss / Milvus / 独立向量数据库**：与轻量化目标直接冲突，N < 2 万时也没有性能收益
- **不要把 decay_weight 预乘进向量矩阵**：会让缓存失效逻辑从「增删触发」退化成「每轮触发」，收益归零
- **不要硬删除未归档的沉底记忆**：休眠唤醒（`decay_weight < 0.1` 被强激活后重回候选池）是本项目的设计特色，硬删会破坏它。归档表保留了这个能力
- **不要在没有归一化 bm25 的情况下上 FTS5**：会静默改变双通道权重的实际含义
- **不要跳过 P3 的阈值重校准**：`SIMILARITY_MID/HIGH` 是空间相关的，换 embedding 维度后沿用旧值等于关掉了激活机制

## 附录 C：与论文工作的关系

本次改动须新建分支，与论文工作完全隔离开。
