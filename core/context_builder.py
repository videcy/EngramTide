"""
Phase 2 — Constitutional AI 上下文组装器。

职责：
- 将检索结果与浮现记忆合并，按记忆类型分组。
- 生成 Constitutional AI 模板需要的四个动态片段。
- Phase 2 新增浮现记忆合并；_format_bucket 同时返回文本和 ID 列表。
"""

from dataclasses import dataclass, field

from core.memory_store import Memory
from config import MAX_MEMORY_CONTEXT_CHARS


# 允许的记忆类型及其归属分类
# type → (分类名, 显示标题)
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


def build_constitutional_memory_context(
    retrieved: list[tuple[Memory, float]],
    max_chars: int = MAX_MEMORY_CONTEXT_CHARS,
    debug: bool = False,
    surfaced: list[Memory] | None = None,
) -> ConstitutionalMemoryContext:
    """
    将检索结果 + 浮现记忆合并，按类型分组，生成 Constitutional AI 上下文。

    Args:
        retrieved: (Memory, score) 列表，已按分数降序。
        max_chars: 每个分类片段的最大字符数。
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
            # 浮现记忆给一个高虚拟分数，以确保在截断时优先保留
            merged_scores[mem.memory_id] = 2.0

    retrieved_by_id: dict[str, tuple[Memory, float]] = {}
    for mem, score in retrieved:
        if mem.memory_id not in merged_scores:
            retrieved_by_id[mem.memory_id] = (mem, score)
            merged_scores[mem.memory_id] = score

    # 按类型分组（合并后）
    buckets: dict[str, list[tuple[Memory, float]]] = {
        "procedural": [],
        "semantic": [],
        "episodic": [],
        "emotional": [],
    }

    # 浮现记忆优先加入
    for mem_id, mem in surfaced_by_id.items():
        category = _TYPE_CATEGORY_MAP.get(mem.type)
        if category is None:
            category = "semantic"
        buckets[category].append((mem, merged_scores[mem_id]))

    # 检索记忆加入（去重后的）
    for mem_id, (mem, score) in retrieved_by_id.items():
        category = _TYPE_CATEGORY_MAP.get(mem.type)
        if category is None:
            category = "semantic"
        buckets[category].append((mem, score))

    # 每个桶内按分数降序
    for cat in buckets:
        buckets[cat].sort(key=lambda x: x[1], reverse=True)

    context = ConstitutionalMemoryContext()

    context.procedural_memories, proc_ids = _format_bucket_with_ids(
        buckets["procedural"], max_chars, debug
    )
    context.semantic_memories, sem_ids = _format_bucket_with_ids(
        buckets["semantic"], max_chars, debug
    )
    context.episodic_memories, epi_ids = _format_bucket_with_ids(
        buckets["episodic"], max_chars, debug
    )
    context.emotional_memories, emo_ids = _format_bucket_with_ids(
        buckets["emotional"], max_chars, debug
    )

    context.included_memory_ids = proc_ids + sem_ids + epi_ids + emo_ids

    return context


def _format_bucket_with_ids(
    items: list[tuple[Memory, float]],
    max_chars: int,
    debug: bool,
) -> tuple[str, list[str]]:
    """
    将一个分类的记忆格式化为文本片段，同时返回实际进入文本的记忆 ID 列表。

    按条目截断，不在单条内容中间截断。
    如果没有条目，返回 ("无", [])。

    Phase 2 合并了原来的 _format_bucket 和 _collect_included_memory_ids。
    """
    if not items:
        return ("无", [])

    lines: list[str] = []
    ids: list[str] = []
    total_chars = 0

    for mem, score in items:
        if debug:
            line = f"- [score={score:.3f}] {mem.content}"
        else:
            line = f"- {mem.content}"

        if total_chars + len(line) > max_chars and lines:
            # 超出限制，不再追加后续条目
            break

        lines.append(line)
        ids.append(mem.memory_id)
        total_chars += len(line)

    return ("\n".join(lines) if lines else "无", ids)
