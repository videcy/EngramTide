"""
Phase 1 MVP — 余弦相似度工具。

处理零向量和维度不一致的边缘情况。
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。

    Args:
        a: 向量 A，dtype=np.float32。
        b: 向量 B，dtype=np.float32。

    Returns:
        float: 余弦相似度，范围 [-1.0, 1.0]。

    Raises:
        ValueError: 向量维度不一致时抛出。
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("向量必须是一维的")

    if a.shape != b.shape:
        raise ValueError(
            f"向量维度不一致：a 是 {a.shape[0]} 维，b 是 {b.shape[0]} 维"
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 任一向量为空或零范数，返回 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    result = float(np.dot(a, b) / (norm_a * norm_b))

    # 浮点精度保护：clip 到 [-1.0, 1.0]
    return max(-1.0, min(1.0, result))
