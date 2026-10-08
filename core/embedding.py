"""
Phase 1 MVP — Embedding API 客户端。

职责：
- 调用 OpenAI-compatible /v1/embeddings 接口。
- 将文本转为 numpy.ndarray(np.float32)。
- 对空文本做保护。
- 请求失败时抛出带上下文的异常。

支持通过环境变量切换 provider：
  EMBEDDING_BASE_URL → 必填，提供 OpenAI-compatible 接口的服务地址
  EMBEDDING_API_KEY  → 必填，独立 embedding 服务凭据
  EMBEDDING_MODEL    → 默认 text-embedding-3-small
"""

import logging
from typing import Any

import httpx
import numpy as np

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# 首次成功 embedding 后确定维度，后续校验一致性
_embedding_dim: int | None = None

# provider 是否支持 OpenAI 规范的可选字段 `dimensions`。
# None = 尚未探测；False = 探测到不支持，此后一律走「全维 + 本地截断」。
_supports_dimensions: bool | None = None


def get_embedding_dim() -> int | None:
    """返回已确定的 embedding 维度，未调用过则返回 None。"""
    return _embedding_dim


def reset_dim_state() -> None:
    """清空维度与 provider 能力探测状态（换 provider / 测试用）。"""
    global _embedding_dim, _supports_dimensions
    _embedding_dim = None
    _supports_dimensions = None


def normalize(vec: np.ndarray) -> np.ndarray:
    """
    L2 归一化。零向量原样返回。

    入库前就归一化，检索时余弦相似度才能退化成一次矩阵乘（见 core/vector_index）。
    MRL 模型截断后也必须重新归一化——截断会改变模长。
    """
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _truncate_to_dim(vec: np.ndarray) -> np.ndarray:
    """
    按 Matryoshka 表征把全维向量截断到 EMBEDDING_DIM，再重新归一化。

    text-embedding-3-* 的前 k 维本身就是一个有效的低维表示，直接截断的检索质量
    损失很小（官方数据：3-small 1536→512，MTEB 检索指标降约 1~2 个点）。
    向量比目标维度还短时不做任何处理——那不是 MRL 截断能解决的问题。
    """
    if EMBEDDING_DIM <= 0 or vec.shape[0] <= EMBEDDING_DIM:
        return normalize(vec)
    return normalize(vec[:EMBEDDING_DIM])


def _build_url() -> str:
    """构建 embedding API 完整 URL。"""
    base = EMBEDDING_BASE_URL.rstrip("/")
    # 不少 provider 文档给的 base_url 本身就带 /v1（如 https://api.siliconflow.cn/v1），
    # 无条件追加会拼出 /v1/v1/embeddings → 404
    if base.endswith("/v1/embeddings"):
        return base
    if base.endswith("/v1"):
        return f"{base}/embeddings"
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

    # provider 忽略了 dimensions（或本来就不支持）→ 本地按 MRL 截断兜底，
    # 保证无论走哪条路径，出口维度都等于 EMBEDDING_DIM。
    vec = _truncate_to_dim(vec)

    # 首次确定维度
    if _embedding_dim is None:
        _embedding_dim = vec.shape[0]
        logger.info("Embedding 维度确定为: %d", _embedding_dim)
    elif vec.shape[0] != _embedding_dim:
        raise ValueError(
            f"Embedding 维度不一致：预期 {_embedding_dim}，实际 {vec.shape[0]}"
        )

    return vec


def _build_payload(text: str, with_dimensions: bool) -> dict[str, Any]:
    """构建请求体；with_dimensions 决定是否带上 `dimensions` 字段。"""
    payload: dict[str, Any] = {"model": EMBEDDING_MODEL, "input": text}
    if with_dimensions:
        payload["dimensions"] = EMBEDDING_DIM
    return payload


async def embed_text(text: str) -> np.ndarray:
    """
    对单条文本调用 embedding API。

    Args:
        text: 输入文本。

    Returns:
        np.ndarray: float32 向量，维度为 EMBEDDING_DIM，且已 L2 归一化
        （归一化是 core/vector_index 把余弦退化成矩阵乘的前提）。

    Raises:
        ValueError: 文本为空时。
        httpx.HTTPError: 网络或 API 错误时。
    """
    text = text.strip()
    if not text:
        raise ValueError("不能对空文本调用 embedding")

    global _supports_dimensions

    url = _build_url()
    headers = _build_headers()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            # `dimensions` 是 OpenAI 规范的可选字段，部分自建 provider 不支持并
            # 直接返回 400。首次请求带上探测一次，失败则记住并永久回退到
            # 「全维 + 本地截断 + 重新归一化」，不再重复浪费一次往返。
            send_dimensions = _supports_dimensions is not False and EMBEDDING_DIM > 0
            response = await client.post(
                url,
                headers=headers,
                json=_build_payload(text, send_dimensions),
            )

            if response.status_code == 400 and send_dimensions:
                logger.info(
                    "Embedding provider 不接受 dimensions 参数（400），"
                    "回退为全维请求 + 本地截断到 %d 维。",
                    EMBEDDING_DIM,
                )
                _supports_dimensions = False
                response = await client.post(
                    url, headers=headers, json=_build_payload(text, False)
                )

            if response.status_code != 200:
                error_summary = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"Embedding API 返回 {response.status_code}: {error_summary}",
                    request=response.request,
                    response=response,
                )

            if send_dimensions and _supports_dimensions is None:
                _supports_dimensions = True

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
