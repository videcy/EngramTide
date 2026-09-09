"""
Phase 4 + P1/P2 — 向量 + 关键词双通道检索器。

职责：
- 向量通道：一次矩阵乘算出全量余弦相似度（core.vector_index），可与激活通道共用。
- 关键词通道：优先 FTS5 倒排索引（O(log N) 召回 + bm25 归一化），
  缺失时降级 rapidfuzz 全表模糊匹配，再缺失时降级纯向量。
- 双通道打分：score = (RETRIEVAL_VECTOR_WEIGHT · sim + RETRIEVAL_KEYWORD_WEIGHT · kw) · decay_weight
- 通道关闭或 query_text=None 时退化为纯向量检索（与 Phase 3 逐位一致）。
- 关键词通道不绕过检索过滤（RETRIEVAL_MIN_DECAY / superseded / 无 embedding）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import (
    FTS_CANDIDATE_LIMIT,
    KEYWORD_CHANNEL_ENABLED,
    RETRIEVAL_KEYWORD_WEIGHT,
    RETRIEVAL_MIN_DECAY,
    RETRIEVAL_VECTOR_WEIGHT,
)
from core.memory_store import Memory, fts_available, fts_search, list_active_memories
from core.vector_index import similarity_map

logger = logging.getLogger(__name__)

# safe import：rapidfuzz 缺失时降级纯向量
try:
    from rapidfuzz import fuzz  # noqa: F811

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

_RAPIDFUZZ_WARNED = False


def _warn_rapidfuzz_missing() -> None:
    """一次性 warning：FTS5 与 rapidfuzz 都不可用，关键词通道降级。"""
    global _RAPIDFUZZ_WARNED
    if not _RAPIDFUZZ_WARNED:
        _RAPIDFUZZ_WARNED = True
        logger.warning(
            "FTS5 与 rapidfuzz 均不可用，关键词通道已降级为纯向量检索。"
            "安装: pip install rapidfuzz>=3.0.0"
        )


@dataclass
class RetrievalDetail:
    """单条检索结果的分数分解，供 /debug retrieval 逐条展示（计划 §9.2）。"""

    memory: Memory
    sim: float          # 向量通道余弦相似度
    kw: float | None    # 关键词通道分（通道未生效时为 None）
    score: float        # 最终分（含 decay_weight 乘数）


def retrieve_memories(
    query_embedding: np.ndarray,
    memories: list[Memory] | None = None,
    top_k: int = 5,
    query_text: str | None = None,
    sims: dict[str, float] | None = None,
) -> list[tuple[Memory, float]]:
    """
    双通道检索 Top-K 记忆。

    Args:
        query_embedding: 用户输入的 embedding 向量。
        memories: 待检索的记忆列表。为 None 时自动从数据库加载。
        top_k: 返回条数，默认 5。
        query_text: 用户原始输入文本（Phase 4：驱动关键词通道）。
        sims: 预先算好的 {memory_id: 余弦相似度}。激活通道已经算过时传进来，
            本轮就不必再算第二遍。

    Returns:
        list[tuple[Memory, float]]: (记忆, 分数) 列表，按分数降序。
    """
    return [
        (d.memory, d.score)
        for d in retrieve_memories_detailed(
            query_embedding,
            memories=memories,
            top_k=top_k,
            query_text=query_text,
            sims=sims,
        )
    ]


def retrieve_memories_detailed(
    query_embedding: np.ndarray,
    memories: list[Memory] | None = None,
    top_k: int = 5,
    query_text: str | None = None,
    sims: dict[str, float] | None = None,
) -> list[RetrievalDetail]:
    """
    retrieve_memories 的完整版：返回逐条分数分解（vec / kw / 最终分）。

    打分规则与过滤条件同 retrieve_memories——本函数是唯一实现，
    retrieve_memories 是它的薄包装。
    """
    if memories is None:
        memories = list_active_memories()

    if sims is None:
        sims = similarity_map(query_embedding, memories)

    has_query_text = query_text is not None and len(query_text.strip()) > 0
    use_keyword = KEYWORD_CHANNEL_ENABLED and has_query_text

    # 关键词通道后端：FTS5 倒排索引优先，rapidfuzz 兜底，都没有则退化纯向量。
    #
    # fts_search 返回 None 表示「这条查询 FTS5 处理不了」（索引不可用，或查询里
    # 没有任何 ≥3 字符的检索项，例如只查「东京」这样的两字词）——这与「查得动但
    # 没命中」是两回事，前者必须降级，后者应当如实记 0。
    kw_hits: dict[str, float] = {}
    use_fts = False
    if use_keyword:
        hits = fts_search(query_text, FTS_CANDIDATE_LIMIT) if fts_available() else None
        if hits is not None:
            kw_hits = hits
            use_fts = True
        elif not _HAS_RAPIDFUZZ:
            _warn_rapidfuzz_missing()
            use_keyword = False

    scored: list[RetrievalDetail] = []

    for mem in memories:
        # 跳过无 embedding 的记忆
        if mem.embedding is None:
            continue

        # 跳过被取代的记忆
        if mem.superseded_by is not None:
            continue

        # 跳过衰减过低的记忆
        if mem.decay_weight < RETRIEVAL_MIN_DECAY:
            continue

        sim = sims.get(mem.memory_id, 0.0)

        if use_keyword:
            if use_fts:
                # FTS5 只召回候选，未命中的记 0——关键词通道是「专有名词兜底」，
                # 不是给每条记忆都打一个模糊分。
                kw = kw_hits.get(mem.memory_id, 0.0)
            else:
                kw = fuzz.partial_ratio(query_text, mem.content) / 100.0
            score = (
                RETRIEVAL_VECTOR_WEIGHT * sim + RETRIEVAL_KEYWORD_WEIGHT * kw
            ) * mem.decay_weight
        else:
            # 纯向量：与 Phase 3 逐位一致
            kw = None
            score = sim * mem.decay_weight

        scored.append(RetrievalDetail(memory=mem, sim=sim, kw=kw, score=score))

    # 按分数降序排序
    scored.sort(key=lambda d: d.score, reverse=True)

    # 截取 Top-K
    return scored[:top_k]
