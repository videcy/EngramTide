"""
Phase 1 MVP — 向量检索器。

职责：
- 从 memory store 获取所有 active memories。
- 对每条记忆计算余弦相似度。
- 按分数排序返回 Top-K。
"""

from __future__ import annotations

import numpy as np

from config import RETRIEVAL_MIN_DECAY
from core.memory_store import Memory, list_active_memories
from utils.similarity import cosine_similarity


def retrieve_memories(
    query_embedding: np.ndarray,
    memories: list[Memory] | None = None,
    top_k: int = 5,
) -> list[tuple[Memory, float]]:
    """
    向量检索 Top-K 记忆。

    Args:
        query_embedding: 用户输入的 embedding 向量。
        memories: 待检索的记忆列表。为 None 时自动从数据库加载。
        top_k: 返回条数，默认 5。

    Returns:
        list[tuple[Memory, float]]: (记忆, 相似度分数) 列表，按分数降序。
    """
    if memories is None:
        memories = list_active_memories()

    scored: list[tuple[Memory, float]] = []

    for mem in memories:
        # 跳过无 embedding 的记忆
        if mem.embedding is None:
            continue

        # 跳过被取代的记忆
        if mem.superseded_by is not None:
            continue

        # 跳过衰减过低的记忆（Phase 2：< RETRIEVAL_MIN_DECAY）
        if mem.decay_weight < RETRIEVAL_MIN_DECAY:
            continue

        sim = cosine_similarity(query_embedding, mem.embedding)
        score = sim * mem.decay_weight
        scored.append((mem, score))

    # 按分数降序排序
    scored.sort(key=lambda x: x[1], reverse=True)

    # 截取 Top-K
    top_results = scored[:top_k]

    return top_results
