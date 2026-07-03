"""
Phase 1 MVP — Embedding API 客户端。

职责：
- 调用 OpenAI-compatible /v1/embeddings 接口。
- 将文本转为 numpy.ndarray(np.float32)。
- 对空文本做保护。
- 请求失败时抛出带上下文的异常。

支持通过环境变量切换 provider：
  EMBEDDING_BASE_URL → 默认沿用 DEEPSEEK_BASE_URL
  EMBEDDING_API_KEY  → 默认沿用 DEEPSEEK_API_KEY
  EMBEDDING_MODEL    → 默认 text-embedding-3-small
"""

import logging
from typing import Any

import httpx
import numpy as np

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# 首次成功 embedding 后确定维度，后续校验一致性
_embedding_dim: int | None = None


def get_embedding_dim() -> int | None:
    """返回已确定的 embedding 维度，未调用过则返回 None。"""
    return _embedding_dim


def _build_url() -> str:
    """构建 embedding API 完整 URL。"""
    base = EMBEDDING_BASE_URL.rstrip("/")
    return f"{base}/v1/embeddings"


def _build_headers() -> dict[str, str]:
    """构建请求头。"""
    return {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }


def _parse_embedding_response(data: dict[str, Any]) -> np.ndarray:
    """从 API 响应中提取 embedding 向量。"""
    global _embedding_dim

    if "data" not in data or len(data["data"]) == 0:
        raise ValueError(f"Embedding API 响应中没有 data 字段: {data}")

    item = data["data"][0]
    if "embedding" not in item:
        raise ValueError(f"Embedding API 响应中没有 embedding 字段: {item}")

    vec = np.array(item["embedding"], dtype=np.float32)

    # 首次确定维度
    if _embedding_dim is None:
        _embedding_dim = vec.shape[0]
        logger.info("Embedding 维度确定为: %d", _embedding_dim)
    elif vec.shape[0] != _embedding_dim:
        raise ValueError(
            f"Embedding 维度不一致：预期 {_embedding_dim}，实际 {vec.shape[0]}"
        )

    return vec


async def embed_text(text: str) -> np.ndarray:
    """
    对单条文本调用 embedding API。

    Args:
        text: 输入文本。

    Returns:
        np.ndarray: float32 向量。

    Raises:
        ValueError: 文本为空时。
        httpx.HTTPError: 网络或 API 错误时。
    """
    text = text.strip()
    if not text:
        raise ValueError("不能对空文本调用 embedding")

    url = _build_url()
    headers = _build_headers()
    payload: dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                error_summary = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"Embedding API 返回 {response.status_code}: {error_summary}",
                    request=response.request,
                    response=response,
                )

            data = response.json()
            return _parse_embedding_response(data)

    except httpx.TimeoutException:
        raise httpx.TimeoutException(
            f"Embedding API 请求超时（{REQUEST_TIMEOUT_SECONDS}s）"
        )
    except httpx.HTTPStatusError:
        raise
    except httpx.RequestError as e:
        raise httpx.RequestError(f"Embedding API 请求失败: {e}") from e


async def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    对多条文本调用 embedding API。

    Phase 1 逐条调用；后续可优化为批量。

    Args:
        texts: 文本列表。

    Returns:
        list[np.ndarray]: 向量列表，与输入一一对应。
    """
    results: list[np.ndarray] = []
    for text in texts:
        vec = await embed_text(text)
        results.append(vec)
    return results
