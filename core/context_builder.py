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

from core.memory_store import Memory
from config import MAX_CONTEXT_TOKENS
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
) -> ConstitutionalMemoryContext:
    """
    将检索结果 + 浮现记忆合并，按类型分组，生成 Constitutional AI 上下文。

    Args:
        retrieved: (Memory, score) 列表，已按分数降序。
        max_tokens: 记忆上下文的总 token 预算（共享，非每桶独立）。
        debug: 是否在内容中包含分数信息。
        surfaced: Phase 2 浮现记忆列表（可选）。与检索合并，浮现优先。

    Returns:
        ConstitutionalMemoryContext: 四个动态片段。
    """
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

    # 共享 token 预算（用可变列表传递）
    budget = [max_tokens]

    # 处理顺序：procedural → semantic → episodic → emotional
    # procedural 在预算内永远最先收录（浮现 R1 保底）
    proc_text, proc_ids, proc_tokens = _format_bucket_with_ids(
        buckets["procedural"], budget, debug
    )
    sem_text, sem_ids, sem_tokens = _format_bucket_with_ids(
        buckets["semantic"], budget, debug
    )
    epi_text, epi_ids, epi_tokens = _format_bucket_with_ids(
        buckets["episodic"], budget, debug
    )
    emo_text, emo_ids, emo_tokens = _format_bucket_with_ids(
        buckets["emotional"], budget, debug
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
