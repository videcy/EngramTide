"""
P1 — 进程内向量矩阵缓存。

为什么不上 faiss / HNSW：
    N < 2~5 万时，L2 归一化 + numpy 矩阵乘的暴力检索比 HNSW 更快，且不需要额外的
    索引内存、持久化与增删同步。ANN 带来的全是「变重」。真正缺索引的是关键词通道
    （见 core/memory_store.fts_search）。active 记忆越过 ANN_BACKEND_THRESHOLD 时
    这里会提示切换 sqlite-vec，在那之前保持暴力矩阵乘。

核心思路：
    cosine(a, b) = â · b̂。只要向量在入库前就做了 L2 归一化（core.embedding 负责），
    整轮的相似度计算就退化成一次 (N, D) @ (D,) 的 BLAS 矩阵乘——从 2N 次 Python
    层函数调用降到 1 次。

缓存失效：
    只跟 memory_store.data_version() 走，即只在**集合成员变化**（插入/删除/
    supersede/归档）时重建。decay_weight 变化不碰矩阵——权重是打分时才乘上去的
    标量，不预乘进矩阵。这一点让失效逻辑保持干净；一旦把权重预乘进去，缓存就得
    每轮重建，收益归零。
"""

from __future__ import annotations

import logging

import numpy as np

from config import VECTOR_MATMUL_BLOCK
from core.memory_store import Memory, data_version

logger = logging.getLogger(__name__)

_ANN_HINT_SHOWN = False


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。零向量保持全零（其相似度恒为 0，与 cosine_similarity 一致）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class VectorCache:
    """active memories 的 (N, D) 归一化矩阵，行序与 self.ids 一一对应。"""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._key: tuple | None = None

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @staticmethod
    def _cache_key(memories: list[Memory]) -> tuple:
        """
        缓存键 = 数据版本号 + 传入集合的指纹。

        只看版本号不够：调用方可能传入一个与库无关的子集或临时列表（测试、
        api 的自定义候选池），此时版本号没变但矩阵内容应该完全不同。
        指纹是 N 个 str 的 hash——Python 缓存字符串哈希，N=10000 也在几百微秒内，
        相对于它守住的一次全量重建可以忽略。
        """
        return (
            data_version(),
            len(memories),
            hash(tuple(m.memory_id for m in memories)),
        )

    def is_fresh(self, memories: list[Memory]) -> bool:
        return self._key is not None and self._key == self._cache_key(memories)

    def invalidate(self) -> None:
        self._key = None
        self._matrix = None
        self.ids = []

    def rebuild(self, memories: list[Memory]) -> None:
        """
        用给定记忆列表重建矩阵。

        维度不一致的行会被剔除：混维只会让 np.vstack 抛异常或让相似度变成垃圾值，
        剔除 + 一条 warning 比两者都好（正经的修法是跑 rebuild_embeddings.py）。
        """
        key = self._cache_key(memories)
        rows = [m for m in memories if m.embedding is not None]
        if not rows:
            self.ids, self._matrix, self._key = [], None, key
            return

        dims = {m.embedding.shape[0] for m in rows}
        if len(dims) > 1:
            majority = max(dims, key=lambda d: sum(1 for m in rows if m.embedding.shape[0] == d))
            dropped = [m for m in rows if m.embedding.shape[0] != majority]
            logger.warning(
                "记忆库中存在 %d 种 embedding 维度 %s，已忽略 %d 条非 %d 维的记忆。"
                "请运行 scripts/rebuild_embeddings.py 统一维度。",
                len(dims), sorted(dims), len(dropped), majority,
            )
            rows = [m for m in rows if m.embedding.shape[0] == majority]

        self.ids = [m.memory_id for m in rows]
        matrix = np.vstack([m.embedding for m in rows]).astype(np.float32, copy=False)
        # 入库时已归一化；这里再归一次是为了兼容旧库里未归一化的向量，
        # 代价是每次重建一遍除法（不是每轮），可以忽略。
        self._matrix = _normalize_rows(matrix)
        self._key = key
        _maybe_hint_ann(len(self.ids))

    def similarities(self, query: np.ndarray) -> tuple[list[str], np.ndarray]:
        """返回 (ids, sims)。sims[i] 是 ids[i] 与 query 的余弦相似度。"""
        if self._matrix is None or query is None:
            return [], np.zeros(0, dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self._matrix.shape[1]:
            logger.warning(
                "查询向量 %d 维与记忆矩阵 %d 维不一致，本轮相似度全部按 0 处理。"
                "多半是 EMBEDDING_DIM 改了但没重建全库。",
                q.shape[0], self._matrix.shape[1],
            )
            return self.ids, np.zeros(len(self.ids), dtype=np.float32)

        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        sims = np.empty(self._matrix.shape[0], dtype=np.float32)
        block = max(1, VECTOR_MATMUL_BLOCK)
        for start in range(0, self._matrix.shape[0], block):
            stop = min(start + block, self._matrix.shape[0])
            np.dot(self._matrix[start:stop], q, out=sims[start:stop])

        # 浮点累加可能溢出 [-1, 1] 一点点，与 cosine_similarity 的 clip 行为对齐
        return self.ids, np.clip(sims, -1.0, 1.0, out=sims)


# 进程内单例。测试要隔离时调 reset_cache()。
_cache = VectorCache()


def reset_cache() -> None:
    """丢弃缓存（切库、测试隔离时调用）。"""
    _cache.invalidate()


def similarity_map(
    query_embedding: np.ndarray | None,
    memories: list[Memory],
) -> dict[str, float]:
    """
    一次矩阵乘算出 query 与全部记忆的余弦相似度，返回 {memory_id: sim}。

    激活通道与检索通道共用同一份结果，各自再套自己的阈值/过滤逻辑——
    这是 P1 收益的主体：每轮 2N 次 Python 层 cosine_similarity 调用变成 1 次 BLAS。

    未出现在返回值中的 id 表示「无 embedding / 维度不符」，调用方按 0 处理。
    """
    if query_embedding is None:
        return {}
    if not _cache.is_fresh(memories):
        _cache.rebuild(memories)
    ids, sims = _cache.similarities(query_embedding)
    return {mid: float(s) for mid, s in zip(ids, sims)}


def _maybe_hint_ann(active_count: int) -> None:
    """越过阈值时提示一次切 sqlite-vec；在那之前暴力矩阵乘就是最优解。"""
    global _ANN_HINT_SHOWN
    from config import ANN_BACKEND_THRESHOLD

    if active_count > ANN_BACKEND_THRESHOLD and not _ANN_HINT_SHOWN:
        _ANN_HINT_SHOWN = True
        logger.warning(
            "活跃记忆 %d 条已超过 ANN_BACKEND_THRESHOLD=%d，"
            "建议改用 sqlite-vec（单个扩展文件，无独立服务）作为向量后端。",
            active_count, ANN_BACKEND_THRESHOLD,
        )
