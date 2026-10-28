"""Embedding 端点拼接：base_url 带不带 /v1 都要拼出正确地址。"""

import pytest

import core.embedding as embedding


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/", "https://api.example.com/v1/embeddings"),
        ("https://api.siliconflow.cn/v1", "https://api.siliconflow.cn/v1/embeddings"),
        ("https://api.siliconflow.cn/v1/", "https://api.siliconflow.cn/v1/embeddings"),
        ("https://host/v1/embeddings", "https://host/v1/embeddings"),
    ],
)
def test_build_url_tolerates_v1_suffix(monkeypatch, base, expected):
    monkeypatch.setattr(embedding, "EMBEDDING_BASE_URL", base)
    assert embedding._build_url() == expected
