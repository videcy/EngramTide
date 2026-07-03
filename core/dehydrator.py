"""
Phase 1 MVP — 会话脱水器。

职责：
- 接收本轮会话历史。
- 调用 LLM 进行脱水，输出 JSON 记忆数组。
- 解析 JSON，清洗字段。
- 为每条记忆计算 embedding。
- 构造 Memory 对象。
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEHYDRATE_MAX_ITEMS,
    PROMPTS_DIR,
    REQUEST_TIMEOUT_SECONDS,
)
from core.embedding import embed_text
from core.memory_store import Memory

logger = logging.getLogger(__name__)

# 允许的记忆类型
_ALLOWED_TYPES = {"semantic", "episodic", "emotional", "procedural"}

# 加载脱水 prompt 模板
_DEHYDRATE_PROMPT_PATH: Path = PROMPTS_DIR / "dehydrate.txt"


def _parse_bool(value: Any, default: bool = False) -> bool:
    """
    显式解析 LLM 返回的布尔字段。

    bool 原样保留；字符串按 lower 后匹配 true/false/yes/no/1/0。
    其他类型无法确定时返回 default，避免 bool("false") == True 的陷阱。
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        return default

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    return default


def _load_dehydrate_prompt() -> str:
    """加载脱水 prompt 模板。"""
    if _DEHYDRATE_PROMPT_PATH.exists():
        return _DEHYDRATE_PROMPT_PATH.read_text(encoding="utf-8")
    # 内置最小后备
    return (
        "从以下对话中提取有用信息，输出 JSON 数组。\n"
        "每条记忆: {{\"content\": \"...\", \"type\": \"semantic\", "
        "\"valence\": 0.0, \"arousal\": 0.0, \"unresolved\": false, \"tags\": []}}\n"
        "最多 {max_items} 条。"
    )


def _format_conversation(messages: list[dict[str, str]]) -> str:
    """将对话历史格式化为可读文本。"""
    lines: list[str] = []
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    从 LLM 输出中提取第一个 JSON 数组。

    先尝试直接解析全文；失败后尝试用正则截取 [...] 段。
    """
    text = text.strip()

    # 尝试 1: 直接解析
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试 2: 正则截取第一个 JSON 数组
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 输出中提取 JSON 数组: {text[:300]}")


def _clean_memory_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    清洗单条记忆的字段。

    返回 None 表示该条应被丢弃。
    """
    content = str(item.get("content", "")).strip()
    if not content:
        return None  # 空 content 丢弃

    mem_type = str(item.get("type", "semantic")).strip().lower()
    if mem_type not in _ALLOWED_TYPES:
        mem_type = "semantic"  # 未知类型降级

    # valence: clamp [-1.0, 1.0]
    try:
        valence = float(item.get("valence", 0.0))
    except (TypeError, ValueError):
        valence = 0.0
    valence = max(-1.0, min(1.0, valence))

    # arousal: clamp [0.0, 1.0]
    try:
        arousal = float(item.get("arousal", 0.0))
    except (TypeError, ValueError):
        arousal = 0.0
    arousal = max(0.0, min(1.0, arousal))

    # unresolved
    unresolved = _parse_bool(item.get("unresolved", False))

    # tags
    tags = item.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()]

    return {
        "content": content,
        "type": mem_type,
        "valence": valence,
        "arousal": arousal,
        "unresolved": unresolved,
        "tags": tags,
    }


async def dehydrate_conversation(
    conversation: list[dict[str, str]],
    source_conv_id: str,
) -> list[Memory]:
    """
    将会话脱水为记忆列表。

    Args:
        conversation: 对话历史，每个元素 {"role": "user"/"assistant", "content": "..."}。
        source_conv_id: 本次会话 ID。

    Returns:
        list[Memory]: 提取出的记忆对象列表。
    """
    if not conversation:
        return []

    prompt_template = _load_dehydrate_prompt()
    conversation_text = _format_conversation(conversation)

    system_prompt = prompt_template.replace("{max_items}", str(DEHYDRATE_MAX_ITEMS))

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请提取以下对话中的记忆：\n\n{conversation_text}"},
        ],
        "temperature": 0.3,  # 低温度，提高 JSON 输出一致性
        "max_tokens": 2048,
    }

    logger.info("正在调用脱水 LLM...")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                error_summary = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"脱水 API 返回 {response.status_code}: {error_summary}",
                    request=response.request,
                    response=response,
                )

            data = response.json()
            raw_output = data["choices"][0]["message"]["content"]
            logger.debug("脱水 LLM 原始输出:\n%s", raw_output)

    except httpx.TimeoutException:
        raise httpx.TimeoutException(
            f"脱水 API 请求超时（{REQUEST_TIMEOUT_SECONDS}s）"
        )
    except httpx.HTTPStatusError:
        raise
    except httpx.RequestError as e:
        raise httpx.RequestError(f"脱水 API 请求失败: {e}") from e

    # 解析 JSON
    raw_items = _extract_json_array(raw_output)

    # 如果 LLM 返回空数组，不写入记忆
    if not raw_items:
        logger.info("脱水 LLM 返回空数组，无新记忆生成。")
        return []

    # 清洗并构造 Memory 对象
    memories: list[Memory] = []
    for item in raw_items:
        cleaned = _clean_memory_item(item)
        if cleaned is None:
            continue

        # 计算 embedding
        try:
            embedding = await embed_text(cleaned["content"])
        except Exception as e:
            logger.warning("记忆 embedding 失败，跳过该条: %s", e)
            continue

        memory = Memory(
            memory_id=str(uuid.uuid4()),
            type=cleaned["type"],
            content=cleaned["content"],
            valence=cleaned["valence"],
            arousal=cleaned["arousal"],
            embedding=embedding,
            source_conv_id=source_conv_id,
            unresolved=cleaned["unresolved"],
            tags=cleaned["tags"],
        )
        memories.append(memory)

    logger.info("脱水完成：%d 条原始 → %d 条有效记忆", len(raw_items), len(memories))
    return memories
