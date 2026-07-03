"""
测试余弦相似度：相同向量、正交向量、零向量、维度不一致。
"""

import numpy as np
import pytest

from utils.similarity import cosine_similarity


def test_identical_vectors():
    """相同向量返回接近 1.0。"""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = cosine_similarity(v, v)
    assert pytest.approx(result, abs=1e-6) == 1.0


def test_orthogonal_vectors():
    """正交向量返回接近 0.0。"""
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = cosine_similarity(a, b)
    assert pytest.approx(result, abs=1e-6) == 0.0


def test_opposite_vectors():
    """完全相反的向量返回接近 -1.0。"""
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    result = cosine_similarity(a, b)
    assert pytest.approx(result, abs=1e-6) == -1.0


def test_zero_vector():
    """零向量返回 0.0。"""
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = cosine_similarity(a, b)
    assert result == 0.0


def test_both_zero_vectors():
    """两个零向量返回 0.0。"""
    a = np.zeros(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    result = cosine_similarity(a, b)
    assert result == 0.0


def test_dimension_mismatch():
    """维度不一致抛出 ValueError。"""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="维度不一致"):
        cosine_similarity(a, b)


def test_not_1d():
    """非一维向量抛出 ValueError。"""
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_similar_vectors_high_score():
    """相似但不完全相同的向量应返回高分（> 0.9）。"""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.1, 2.1, 2.9, 4.0], dtype=np.float32)
    result = cosine_similarity(a, b)
    assert result > 0.99
