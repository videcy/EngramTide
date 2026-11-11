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
import uuid
from dataclasses import dataclass, replace

import config
from config import (
    EMOTIONAL_REINFORCE_BOOST,
    EMOTIONAL_REINFORCE_THRESHOLD,
    EPISODIC_DEDUP_THRESHOLD,
    EPISODIC_REINFORCE_BOOST,
    PROCEDURAL_DEDUP_THRESHOLD,
    SEMANTIC_OVERRIDE_THRESHOLD,
)
from core.conflict import (
    Candidate,
    Relation,
    find_conflict_candidates,
    judge_by_rules,
    judge_relations,
)
from core.embedding import embed_text
from core.memory_store import (
    Memory,
    clear_unresolved,
    get_memory,
    insert_link,
    insert_memory,
    list_active_memories,
    log_access_event,
    mark_accessed,
    mark_superseded,
    reinforce_memory,
    scale_decay_weight,
    supersede_and_archive,
)
from utils.similarity import cosine_similarity
from utils.time_utils import parse_db_timestamp, utc_now

logger = logging.getLogger(__name__)


@dataclass
class WriteReport:
    """一次写入管线的摘要，供退出日志与测试断言。"""

    inserted: int = 0     # 实际新插入条数
    superseded: int = 0   # 被标记 superseded 的旧记忆条数
    reinforced: int = 0   # 被强化的旧 emotional 条数
    deduped: int = 0      # 被去重拦截的 procedural 条数
    failed: int = 0       # 单条失败跳过条数
    updated: int = 0      # 冲突判定：新记忆更新/更正了旧记忆（含 followup）
    refined: int = 0      # 冲突判定：新记忆补充了旧记忆
    conflicted: int = 0   # 冲突判定：挂起待确认的冲突对数
    damped: int = 0       # emotional 态度反转，旧记忆被降权的条数


# 最近一次 write_memories 的判定明细（每次调用开头清空），供 /debug conflicts
LAST_JUDGMENTS: list[str] = []


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
    LAST_JUDGMENTS.clear()
    if not new_memories:
        return WriteReport()

    if config.CONFLICT_DETECT_ENABLED:
        return await _write_with_conflict(new_memories)

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


# ── 冲突感知写入管线 ──────────────────────────────────────

_VERBS = {
    "duplicate": "重复",
    "refine": "补充",
    "update": "覆盖",
    "contradict": "挂起",
    "unrelated": "无关",
}


def _clip(text: str, n: int = 30) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _log_judgment(
    rel: Relation,
    new_mem: Memory,
    old: Memory,
    *,
    verb: str | None = None,
    tag: str | None = None,
    note: str | None = None,
) -> None:
    """追加一行判定明细：[relation|sim=0.71|llm] 新 → 覆盖：旧（理由）。"""
    tag = tag or rel.relation
    verb = verb or _VERBS.get(rel.relation, rel.relation)
    arrow = "⇄" if verb == "挂起" else "→"
    reason = note if note is not None else rel.reason
    tail = f"（{reason}）" if reason else ""
    LAST_JUDGMENTS.append(
        f"[{tag}|sim={rel.similarity:.2f}|{rel.judged_by}] "
        f"{_clip(new_mem.content)} {arrow} {verb}：{_clip(old.content)}{tail}"
    )


def _created_at(mem: Memory):
    """created_at 转 aware datetime；空串或无法解析视为当前 UTC 时间。"""
    if not mem.created_at:
        return utc_now()
    try:
        return parse_db_timestamp(mem.created_at)
    except ValueError:
        return utc_now()


def _guard_failure(new_mem: Memory, old: Memory, *, skip_access: bool = False) -> str | None:
    """覆盖旧记忆前的三条守卫；全部通过返回 None，否则返回失败原因。"""
    if _created_at(new_mem) < _created_at(old):
        return "新记忆的时间早于旧记忆"
    if old.origin == "remember" and new_mem.origin != "remember":
        return "旧记忆来自显式 remember，脱水/导入不能自动覆盖"
    if not skip_access and old.access_count >= config.CONFLICT_PROTECT_ACCESS:
        return f"旧记忆 access_count={old.access_count}，受保护"
    return None


def _link(new_id: str, old_id: str, rel: Relation, relation: str, status: str,
          subtype: str | None) -> None:
    insert_link(
        new_id,
        old_id,
        relation,
        judged_by=rel.judged_by,
        status=status,
        subtype=subtype,
        similarity=rel.similarity,
        reason=rel.reason,
    )


class _BatchState:
    """一次批量写入内的状态：原新记忆 id → 实际入库 id（被吸收的不在内）。"""

    def __init__(self, batch_ids: set[str]):
        self.batch_ids = batch_ids
        self.inserted: dict[str, str] = {}

    def resolve_old(self, old_id: str) -> Memory | None:
        """把关系里的 old 解析成库里当前仍有效的记忆；不存在/已被覆盖返回 None。"""
        if old_id in self.batch_ids:
            old_id = self.inserted.get(old_id)
            if old_id is None:
                return None
        mem = get_memory(old_id)
        if mem is None or mem.superseded_by is not None:
            return None
        return mem


async def _write_with_conflict(new_memories: list[Memory]) -> WriteReport:
    report = WriteReport()
    pool = list_active_memories()

    # 1. 召回：每条新记忆的候选池 = 旧库 + 本批中排在它前面的新记忆
    cands_per_new: list[list[Candidate]] = []
    for i, new_mem in enumerate(new_memories):
        try:
            cands = find_conflict_candidates(new_mem, pool + new_memories[:i])
        except Exception as e:
            logger.warning("召回冲突候选失败 %s: %s", new_mem.memory_id, e)
            cands = []
        cands_per_new.append(cands)

    # 2. 整批一次判定；任何异常都退回规则，不丢新记忆
    all_cands = [c for cands in cands_per_new for c in cands]
    relations: list[Relation] = []
    if all_cands:
        try:
            relations = list(await judge_relations(all_cands))
            if len(relations) != len(all_cands):
                raise ValueError("判定结果数量与候选不一致")
        except Exception as e:
            logger.warning("关系判定失败，退回规则: %s", e)
            relations = []
        if not relations:
            relations = [_safe_rule(c) for c in all_cands]

    # 3. 逐条处置
    state = _BatchState({m.memory_id for m in new_memories})
    offset = 0
    for new_mem, cands in zip(new_memories, cands_per_new):
        pairs = list(zip(cands, relations[offset:offset + len(cands)]))
        offset += len(cands)
        try:
            await _process_one(new_mem, pairs, state, report)
        except Exception as e:
            logger.warning("写入记忆 %s 失败: %s", new_mem.memory_id, e)
            report.failed += 1
    return report


def _safe_rule(c: Candidate) -> Relation:
    try:
        return judge_by_rules(c)
    except Exception:
        return Relation(
            new_id=c.new.memory_id, old_id=c.old.memory_id, relation="unrelated",
            similarity=c.similarity, judged_by="rule", reason="规则判定失败",
        )


def _insert(new_mem: Memory, state: _BatchState, report: WriteReport) -> str:
    insert_memory(new_mem)
    report.inserted += 1
    state.inserted[new_mem.memory_id] = new_mem.memory_id
    return new_mem.memory_id


async def _process_one(
    new_mem: Memory,
    pairs: list[tuple[Candidate, Relation]],
    state: _BatchState,
    report: WriteReport,
) -> None:
    orig_id = new_mem.memory_id

    # 过滤：old 已不在库（被吸收/已覆盖）的关系忽略；unrelated 只记日志
    actionable: list[tuple[Relation, Memory]] = []
    for cand, rel in pairs:
        old = state.resolve_old(rel.old_id)
        if old is None:
            _log_judgment(rel, new_mem, cand.old, note="忽略：旧记忆已不在库中")
            continue
        if rel.relation == "unrelated":
            _log_judgment(rel, new_mem, old)
            continue
        if rel.relation == "duplicate" and (
            old.type != new_mem.type
            or (new_mem.type == "episodic" and not config.EPISODIC_DEDUP_ENABLED)
        ):
            _log_judgment(rel, new_mem, old, note="重复但不吸收，按无关处理")
            continue
        actionable.append((rel, old))

    if not actionable:
        _insert(new_mem, state, report)
        return

    # duplicate 优先：取相似度最高的一条，其余关系忽略
    dups = [x for x in actionable if x[0].relation == "duplicate"]
    if dups:
        rel, old = max(dups, key=lambda x: x[0].similarity)
        _log_judgment(rel, new_mem, old)
        _handle_duplicate(new_mem, old, report)
        return

    actionable.sort(key=lambda x: x[0].similarity, reverse=True)
    nt = new_mem.type

    # refine 合并：semantic/procedural 取第一条带 merged 的
    merge_pair = None
    final_mem = new_mem
    merge_guard_fail: str | None = None
    for rel, old in actionable:
        if (rel.relation == "refine" and nt in ("semantic", "procedural")
                and old.type == nt and rel.merged and rel.merged.strip()):
            merge_pair = (rel, old)
            break
    if merge_pair is not None:
        rel, old = merge_pair
        merge_guard_fail = _guard_failure(new_mem, old, skip_access=(nt == "procedural"))
        if merge_guard_fail is None:
            try:
                emb = await embed_text(rel.merged)
                final_mem = replace(
                    new_mem, memory_id=str(uuid.uuid4()), content=rel.merged, embedding=emb,
                )
            except Exception as e:
                logger.warning("refine 合并文本 embedding 失败，原样插入: %s", e)
                merge_pair = None
                _log_judgment(rel, new_mem, old, note="合并 embedding 失败，原文插入")
        # 守卫失败：原文插入 + pending contradict，在下面逐对处置里处理

    need_insert = any(
        not (rel.relation == "refine" and nt == "emotional") for rel, _ in actionable
    )
    final_id: str | None = None
    if need_insert:
        insert_memory(final_mem)
        report.inserted += 1
        final_id = final_mem.memory_id
        state.inserted[orig_id] = final_id

    for rel, old in actionable:
        if merge_pair is not None and rel is merge_pair[0]:
            if merge_guard_fail is not None:
                _pending(final_id, old, rel, new_mem, "contradict", merge_guard_fail, report)
            else:
                n = supersede_and_archive([old.memory_id], final_id)
                _link(final_id, old.memory_id, rel, "refine", "resolved", rel.subtype)
                _log_judgment(rel, new_mem, old, verb="合并覆盖")
                report.refined += 1
                report.superseded += n
            continue
        _apply_pair(new_mem, final_id, rel, old, report)


def _handle_duplicate(new_mem: Memory, old: Memory, report: WriteReport) -> None:
    """duplicate：semantic/procedural 只计访问；emotional/episodic 沿用强化。"""
    t = new_mem.type
    if t in ("semantic", "procedural"):
        mark_accessed([old.memory_id])
        if config.ACCESS_LOG_ENABLED:
            log_access_event(old.memory_id, "dedup")
        report.deduped += 1
    elif t == "emotional":
        new_weight = min(1.0, old.decay_weight + EMOTIONAL_REINFORCE_BOOST)
        reinforce_memory(
            old.memory_id, new_weight, max(old.arousal, new_mem.arousal),
            list(set(old.tags + new_mem.tags)),
        )
        report.reinforced += 1
    else:
        new_weight = min(1.0, old.decay_weight + EPISODIC_REINFORCE_BOOST)
        reinforce_memory(
            old.memory_id, new_weight, old.arousal, list(set(old.tags + new_mem.tags)),
        )
        report.reinforced += 1


def _pending(
    final_id: str, old: Memory, rel: Relation, new_mem: Memory,
    relation: str, reason: str | None, report: WriteReport, tag: str | None = None,
) -> None:
    """两条都留，写 pending 关系。reason 为守卫失败原因或判定理由。"""
    _link(final_id, old.memory_id, rel, relation, "pending", rel.subtype)
    if tag is None and rel.relation == "contradict" and reason == rel.reason:
        _log_judgment(rel, new_mem, old, verb="挂起")
    else:
        _log_judgment(rel, new_mem, old, verb="挂起", tag=tag or f"{rel.relation}→pending",
                      note=reason)
    report.conflicted += 1


def _overwrite(
    new_mem: Memory, final_id: str, rel: Relation, old: Memory, relation: str,
    subtype: str | None, report: WriteReport,
) -> None:
    """受守卫约束的覆盖归档；守卫失败则挂起。"""
    reason = _guard_failure(new_mem, old, skip_access=(new_mem.type == "procedural"))
    if reason:
        _pending(final_id, old, rel, new_mem, relation, reason, report)
        return
    n = supersede_and_archive([old.memory_id], final_id)
    _link(final_id, old.memory_id, rel, relation, "resolved", subtype)
    _log_judgment(rel, new_mem, old, verb="覆盖")
    report.updated += 1
    report.superseded += n


def _apply_pair(
    new_mem: Memory, final_id: str | None, rel: Relation, old: Memory, report: WriteReport,
) -> None:
    """按处置矩阵处理单个 (新, 旧) 关系；新记忆已按需插入（final_id）。"""
    nt, ot, r = new_mem.type, old.type, rel.relation
    if r == "contradict" and nt in ("procedural", "emotional"):
        r = "update"

    # 跨类型
    if nt == "semantic" and ot == "episodic":
        _link(final_id, old.memory_id, rel, r, "resolved", rel.subtype)
        _log_judgment(rel, new_mem, old, note=f"跨类型只记关系：{rel.reason or ''}")
        return
    if nt == "episodic" and ot == "semantic":
        if r == "update":
            _overwrite(new_mem, final_id, rel, old, "update", None, report)
        elif r == "refine":
            _link(final_id, old.memory_id, rel, "refine", "resolved", None)
            _log_judgment(rel, new_mem, old)
            report.refined += 1
        else:  # contradict
            _pending(final_id, old, rel, new_mem, "contradict", rel.reason, report)
        return

    if r == "refine":
        if nt == "emotional":
            new_weight = min(1.0, old.decay_weight + EMOTIONAL_REINFORCE_BOOST)
            reinforce_memory(
                old.memory_id, new_weight, max(old.arousal, new_mem.arousal),
                list(set(old.tags + new_mem.tags)),
            )
            report.reinforced += 1
        else:
            _link(final_id, old.memory_id, rel, "refine", "resolved", None)
        _log_judgment(rel, new_mem, old)
        report.refined += 1
    elif r == "update":
        if nt == "emotional":
            scale_decay_weight(old.memory_id, config.EMOTIONAL_FLIP_DAMPING)
            _link(final_id, old.memory_id, rel, "update", "resolved", "flip")
            _log_judgment(rel, new_mem, old, verb="降权")
            report.damped += 1
        elif nt == "episodic" and rel.subtype == "followup":
            clear_unresolved(old.memory_id)
            _link(final_id, old.memory_id, rel, "update", "resolved", "followup")
            _log_judgment(rel, new_mem, old, verb="进展")
            report.updated += 1
        else:
            subtype = "correction" if nt == "episodic" else rel.subtype
            _overwrite(new_mem, final_id, rel, old, "update", subtype, report)
    else:  # contradict：semantic / episodic
        _pending(final_id, old, rel, new_mem, "contradict", rel.reason, report)
