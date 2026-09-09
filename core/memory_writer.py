"""
Phase 2 — 类型感知写入管线。

管线顺序（对每条新记忆）：
  semantic   → 覆盖检测：insert 新记忆，再把高相似旧 semantic 标记 superseded_by
  emotional  → 强化检测：命中则更新旧记忆（不 insert），未命中则 insert
  procedural → 去重检测：近重复则只 mark_accessed 旧记忆（不 insert），否则 insert
  episodic   → 近重复检测：命中则强化已有记忆（不 insert），否则 insert

P4-①：episodic 原先是四类里唯一无条件 insert 的类型——而它恰恰数量最大、衰减最快，
是记忆库「只增不减」的主要来源。补上门槛后，写入速率不再是纯增量。
"""

import logging
from dataclasses import dataclass

import config
from config import (
    EMOTIONAL_REINFORCE_BOOST,
    EMOTIONAL_REINFORCE_THRESHOLD,
    EPISODIC_DEDUP_THRESHOLD,
    EPISODIC_REINFORCE_BOOST,
    PROCEDURAL_DEDUP_THRESHOLD,
    SEMANTIC_OVERRIDE_THRESHOLD,
)
from core.memory_store import (
    Memory,
    insert_memory,
    list_active_memories,
    log_access_event,
    mark_accessed,
    mark_superseded,
    reinforce_memory,
)
from utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class WriteReport:
    """一次写入管线的摘要，供退出日志与测试断言。"""

    inserted: int = 0     # 实际新插入条数
    superseded: int = 0   # 被标记 superseded 的旧记忆条数
    reinforced: int = 0   # 被强化的旧 emotional 条数
    deduped: int = 0      # 被去重拦截的 procedural 条数
    failed: int = 0       # 单条失败跳过条数


def _compute_similarity(
    new_embedding,
    old_embedding,
) -> float:
    """
    计算两条 embedding 的余弦相似度。

    任一为 None 返回 0.0（保守：不匹配）。
    """
    if new_embedding is None or old_embedding is None:
        return 0.0
    try:
        return cosine_similarity(new_embedding, old_embedding)
    except Exception:
        return 0.0


async def write_memories(new_memories: list[Memory]) -> WriteReport:
    """
    写入管线入口：对每条新记忆按类型执行差异化写入策略。

    注意：semantic/emotional/procedural 需要与已有记忆做向量相似度比较，
    因此要求 new_memories 中的记忆已携带 embedding。
    无 embedding 的记忆降级为直接 insert（episodic 策略）。
    """
    if not new_memories:
        return WriteReport()

    # 一次性加载所有 active 记忆作为候选池
    existing = list_active_memories()

    # 按类型分组已有记忆（避免重复遍历）
    existing_by_type: dict[str, list[Memory]] = {}
    for mem in existing:
        existing_by_type.setdefault(mem.type, []).append(mem)

    report = WriteReport()

    for new_mem in new_memories:
        try:
            if new_mem.type == "semantic":
                _handle_semantic(new_mem, existing_by_type.get("semantic", []), report)
            elif new_mem.type == "emotional":
                _handle_emotional(new_mem, existing_by_type.get("emotional", []), report)
            elif new_mem.type == "procedural":
                _handle_procedural(new_mem, existing_by_type.get("procedural", []), report)
            elif new_mem.type == "episodic":
                _handle_episodic(new_mem, existing_by_type.get("episodic", []), report)
            else:
                # 未知类型 → 直接插入（fail-safe：宁可多存，不丢数据）
                insert_memory(new_mem)
                report.inserted += 1
        except Exception as e:
            logger.warning("写入记忆 %s 失败: %s", new_mem.memory_id, e)
            report.failed += 1

    return report


def _handle_semantic(
    new_mem: Memory,
    existing_semantic: list[Memory],
    report: WriteReport,
) -> None:
    """
    semantic 覆盖检测：
    1. 先 insert 新记忆。
    2. 对已有 semantic 记忆逐一计算相似度。
    3. 相似度 >= SEMANTIC_OVERRIDE_THRESHOLD 的旧记忆标记 superseded_by。
    """
    insert_memory(new_mem)
    report.inserted += 1

    if new_mem.embedding is None:
        return

    superseded_ids: list[str] = []
    for old_mem in existing_semantic:
        if old_mem.superseded_by is not None:
            continue
        sim = _compute_similarity(new_mem.embedding, old_mem.embedding)
        if sim >= SEMANTIC_OVERRIDE_THRESHOLD:
            superseded_ids.append(old_mem.memory_id)

    if superseded_ids:
        mark_superseded(superseded_ids, new_mem.memory_id)
        report.superseded += len(superseded_ids)


def _handle_emotional(
    new_mem: Memory,
    existing_emotional: list[Memory],
    report: WriteReport,
) -> None:
    """
    emotional 强化检测：
    1. 对已有 emotional 记忆逐一计算相似度。
    2. 命中（相似度 >= EMOTIONAL_REINFORCE_THRESHOLD）→ 强化旧记忆，不 insert。
    3. 未命中 → 直接 insert。
    """
    if new_mem.embedding is None:
        insert_memory(new_mem)
        report.inserted += 1
        return

    for old_mem in existing_emotional:
        if old_mem.superseded_by is not None:
            continue
        sim = _compute_similarity(new_mem.embedding, old_mem.embedding)
        if sim >= EMOTIONAL_REINFORCE_THRESHOLD:
            # 强化：提升 decay_weight、取 max arousal、合并 tags
            new_weight = min(1.0, old_mem.decay_weight + EMOTIONAL_REINFORCE_BOOST)
            new_arousal = max(old_mem.arousal, new_mem.arousal)
            merged_tags = list(set(old_mem.tags + new_mem.tags))
            reinforce_memory(old_mem.memory_id, new_weight, new_arousal, merged_tags)
            report.reinforced += 1
            return

    # 未命中 → insert
    insert_memory(new_mem)
    report.inserted += 1


def _handle_procedural(
    new_mem: Memory,
    existing_procedural: list[Memory],
    report: WriteReport,
) -> None:
    """
    procedural 去重检测：
    1. 对已有 procedural 记忆逐一计算相似度。
    2. 命中（相似度 >= PROCEDURAL_DEDUP_THRESHOLD）→ mark_accessed 旧记忆，不 insert。
    3. 未命中 → insert。
    """
    if new_mem.embedding is None:
        insert_memory(new_mem)
        report.inserted += 1
        return

    for old_mem in existing_procedural:
        if old_mem.superseded_by is not None:
            continue
        sim = _compute_similarity(new_mem.embedding, old_mem.embedding)
        if sim >= PROCEDURAL_DEDUP_THRESHOLD:
            # 去重：只标记访问，不写入新记忆
            mark_accessed([old_mem.memory_id])
            if config.ACCESS_LOG_ENABLED:
                log_access_event(old_mem.memory_id, "dedup")
            report.deduped += 1
            return

    # 未命中 → insert
    insert_memory(new_mem)
    report.inserted += 1


def _handle_episodic(
    new_mem: Memory,
    existing_episodic: list[Memory],
    report: WriteReport,
) -> None:
    """
    episodic 近重复检测（P4-①）：

    1. 找出最相似的一条已有 episodic。
    2. 相似度 >= EPISODIC_DEDUP_THRESHOLD → 强化它（提权重 + 合并标签 + 计访问），不新增。
    3. 否则 insert。

    ★ 语义权衡：episodic 是情景记忆，理论上两次相似但不同时刻的事件应当各自成条。
      所以这里复用的是 emotional 的**强化**思路而不是 semantic 的**覆盖**思路，
      且阈值定得比 semantic 的 0.85 更高（默认 0.88）——只拦截「同一件事被反复
      脱水出来」的真重复。拦多了就调高阈值，或直接 EPISODIC_DEDUP_ENABLED=false。
    """
    if not config.EPISODIC_DEDUP_ENABLED or new_mem.embedding is None:
        insert_memory(new_mem)
        report.inserted += 1
        return

    best: Memory | None = None
    best_sim = 0.0
    for old_mem in existing_episodic:
        if old_mem.superseded_by is not None:
            continue
        sim = _compute_similarity(new_mem.embedding, old_mem.embedding)
        if sim > best_sim:
            best, best_sim = old_mem, sim

    if best is not None and best_sim >= EPISODIC_DEDUP_THRESHOLD:
        new_weight = min(1.0, best.decay_weight + EPISODIC_REINFORCE_BOOST)
        merged_tags = list(set(best.tags + new_mem.tags))
        reinforce_memory(best.memory_id, new_weight, best.arousal, merged_tags)
        report.reinforced += 1
        return

    insert_memory(new_mem)
    report.inserted += 1
