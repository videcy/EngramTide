"""
Phase 4 — Constitutional AI 上下文组装器（Token 预算版）。

职责：
- 将检索结果与浮现记忆合并，按记忆类型分组。
- 用共享 token 预算（max_tokens）截断，以 estimate_tokens 替换字符计数。
- 截断优先级不变：浮现优先（虚拟分 2.0）→ 检索分数降序；procedural 保底。
- ConstitutionalMemoryContext 新增 est_tokens / dropped_count 字段。
"""

import logging
from dataclasses import dataclass, field

import config
from core.memory_store import Memory
from config import MAX_CONTEXT_TOKENS
from utils.time_utils import parse_db_timestamp
from utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

# 允许的记忆类型及其归属分类
_TYPE_CATEGORY_MAP = {
    "procedural": "procedural",
    "semantic": "semantic",
    "episodic": "episodic",
    "emotional": "emotional",
}


@dataclass
class ConstitutionalMemoryContext:
    """Constitutional AI 的四个动态记忆片段。"""

    procedural_memories: str = "无"
    semantic_memories: str = "无"
    episodic_memories: str = "无"
    emotional_memories: str = "无"
    included_memory_ids: list[str] = field(default_factory=list)
    est_tokens: int = 0       # Phase 4：实际收录的估算 token 数
    dropped_count: int = 0    # Phase 4：因预算被丢弃的条目数


def build_constitutional_memory_context(
    retrieved: list[tuple[Memory, float]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    debug: bool = False,
    surfaced: list[Memory] | None = None,
    exclude_ids: frozenset[str] | set[str] | None = None,
    pending_links: list | None = None,
) -> ConstitutionalMemoryContext:
    """
    将检索结果 + 浮现记忆合并，按类型分组，生成 Constitutional AI 上下文。

    Args:
        retrieved: (Memory, score) 列表，已按分数降序。
        max_tokens: 记忆上下文的总 token 预算（共享，非每桶独立）。
        debug: 是否在内容中包含分数信息。
        surfaced: Phase 2 浮现记忆列表（可选）。与检索合并，浮现优先。
        exclude_ids: 不参与组装的记忆 ID（hook 会话内已注入过的）。在预算截断
            之前排除，不占预算，也不计入 dropped_count。
        pending_links: 挂起的冲突关系（MemoryLink 列表）。None 时内部查询一次；
            CONFLICT_DETECT_ENABLED=False 时不查、不加标注。

    Returns:
        ConstitutionalMemoryContext: 四个动态片段。
    """
    if exclude_ids:
        retrieved = [(m, sc) for m, sc in retrieved if m.memory_id not in exclude_ids]
        if surfaced:
            surfaced = [m for m in surfaced if m.memory_id not in exclude_ids]

    # 合并检索 + 浮现（浮现优先，按 memory_id 去重）
    merged_scores: dict[str, float] = {}
    surfaced_by_id: dict[str, Memory] = {}
    if surfaced:
        for mem in surfaced:
            surfaced_by_id[mem.memory_id] = mem
            merged_scores[mem.memory_id] = 2.0  # 浮现记忆高虚拟分确保优先保留

    retrieved_by_id: dict[str, tuple[Memory, float]] = {}
    for mem, score in retrieved:
        if mem.memory_id not in merged_scores:
            retrieved_by_id[mem.memory_id] = (mem, score)
            merged_scores[mem.memory_id] = score

    # 按类型分组
    buckets: dict[str, list[tuple[Memory, float]]] = {
        "procedural": [],
        "semantic": [],
        "episodic": [],
        "emotional": [],
    }

    for mem_id, mem in surfaced_by_id.items():
        category = _TYPE_CATEGORY_MAP.get(mem.type, "semantic")
        buckets[category].append((mem, merged_scores[mem_id]))

    for mem_id, (mem, score) in retrieved_by_id.items():
        category = _TYPE_CATEGORY_MAP.get(mem.type, "semantic")
        buckets[category].append((mem, score))

    # 每个桶内按分数降序
    for cat in buckets:
        buckets[cat].sort(key=lambda x: x[1], reverse=True)

    # 总条目数（用于 dropped_count）
    total_entries = sum(len(b) for b in buckets.values())

    context = ConstitutionalMemoryContext()

    # 挂起冲突标注（一次查询；只对本次入桶的记忆生成）
    notes = _build_conflict_notes(
        [m for b in buckets.values() for m, _ in b], pending_links
    )

    # 共享 token 预算（用可变列表传递）
    budget = [max_tokens]

    # 处理顺序：procedural → semantic → episodic → emotional
    # procedural 在预算内永远最先收录（浮现 R1 保底）
    proc_text, proc_ids, proc_tokens = _format_bucket_with_ids(
        buckets["procedural"], budget, debug, notes
    )
    sem_text, sem_ids, sem_tokens = _format_bucket_with_ids(
        buckets["semantic"], budget, debug, notes
    )
    epi_text, epi_ids, epi_tokens = _format_bucket_with_ids(
        buckets["episodic"], budget, debug, notes
    )
    emo_text, emo_ids, emo_tokens = _format_bucket_with_ids(
        buckets["emotional"], budget, debug, notes
    )

    context.procedural_memories = proc_text
    context.semantic_memories = sem_text
    context.episodic_memories = epi_text
    context.emotional_memories = emo_text
    context.included_memory_ids = proc_ids + sem_ids + epi_ids + emo_ids
    context.est_tokens = proc_tokens + sem_tokens + epi_tokens + emo_tokens
    context.dropped_count = total_entries - len(context.included_memory_ids)

    # procedural 保底断言：若 procedural 条目未被全部收录，log warning
    expected_proc = len(buckets["procedural"])
    actual_proc = len(proc_ids)
    if actual_proc < expected_proc:
        logger.warning(
            "procedural 记忆 (%d 条) 因 token 预算不足被截断为 %d 条——"
            "写入去重可能失效或预算设置过低",
            expected_proc,
            actual_proc,
        )

    return context


def _format_bucket_with_ids(
    items: list[tuple[Memory, float]],
    budget: list[int],  # [remaining_tokens]，可变，调用方共享
    debug: bool,
    notes: dict[str, str] | None = None,
) -> tuple[str, list[str], int]:
    """
    将一个分类的记忆格式化为文本片段。

    返回 (文本, ID 列表, 消耗的估算 token 数)。

    使用共享 token 预算：每收录一条就从 budget[0] 扣减。
    按条目截断，不在单条内容中间截断。
    如果没有条目或预算已耗尽，返回 ("无", [], 0)。
    """
    if not items:
        return ("无", [], 0)

    remaining = budget[0]
    if remaining <= 0:
        return ("无", [], 0)

    lines: list[str] = []
    ids: list[str] = []
    tokens_used = 0

    for mem, score in items:
        if debug:
            line = f"- [score={score:.3f}] {mem.content}"
        else:
            line = f"- {mem.content}"

        note = notes.get(mem.memory_id) if notes else None
        if note:
            # 标注计入预算；带标注放不下时退回不带标注的原行
            with_note = line + note
            if tokens_used + estimate_tokens(with_note) <= remaining:
                line = with_note

        line_tokens = estimate_tokens(line)

        if tokens_used + line_tokens > remaining:
            # 超出预算，不再追加后续条目（含首条：单条超预算直接丢弃，
            # est_tokens ≤ max_tokens 恒成立，预算不允许透支）
            break

        lines.append(line)
        ids.append(mem.memory_id)
        tokens_used += line_tokens

    budget[0] = remaining - tokens_used
    text = "\n".join(lines) if lines else "无"
    return (text, ids, tokens_used)


_NOTE_SNIPPET_CHARS = 60


def _build_conflict_notes(
    mems: list[Memory],
    pending_links: list | None,
) -> dict[str, str]:
    """
    为处于 pending 冲突关系中的记忆生成标注，返回 {memory_id: 标注文本}。

    对方内容用 get_memory 取，取不到（已归档/删除）就不加标注。
    一条记忆涉及多个 pending 关系时只取最新的一个（pending_links 已按新到旧）。
    """
    if not config.CONFLICT_DETECT_ENABLED or not mems:
        return {}
    if pending_links is None:
        try:
            from core.memory_store import list_pending_links

            pending_links = list_pending_links()
        except Exception as e:
            logger.warning("读取挂起冲突失败，跳过冲突标注: %s", e)
            return {}
    if not pending_links:
        return {}

    from core.memory_store import get_memory

    by_id = {m.memory_id: m for m in mems}
    notes: dict[str, str] = {}
    for link in pending_links:
        for mid, other_id in ((link.src_id, link.dst_id), (link.dst_id, link.src_id)):
            mem = by_id.get(mid)
            if mem is None or mid in notes:
                continue
            try:
                other = get_memory(other_id)
            except Exception:
                other = None
            if other is None:
                continue
            text = other.content.strip().replace("\n", " ")
            if len(text) > _NOTE_SNIPPET_CHARS:
                text = text[:_NOTE_SNIPPET_CHARS] + "…"
            word = "较新" if _is_newer(other, mem) else "较早"
            notes[mid] = (
                f"（与{word}的记录“{text}”不一致，尚未确认；话题相关时可向用户确认）"
            )
    return notes


def _is_newer(a: Memory, b: Memory) -> bool:
    """a 的 created_at 是否晚于 b；解析失败或相等视为否。"""
    try:
        return parse_db_timestamp(a.created_at) > parse_db_timestamp(b.created_at)
    except (ValueError, TypeError):
        return False
