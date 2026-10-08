"""
Phase 1 MVP — 聊天 LLM 客户端。

职责：
- 读取 prompts/system_constitution.txt 模板。
- 将 ConstitutionalMemoryContext 填入 system prompt。
- 调用 OpenAI-compatible /v1/chat/completions 接口生成回复。
- API 失败时抛出清晰错误，不静默失败。
"""

import logging
from typing import Any

import httpx

import config
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    MAX_HISTORY_MESSAGES,
    REQUEST_TIMEOUT_SECONDS,
)
from core.context_builder import ConstitutionalMemoryContext

logger = logging.getLogger(__name__)

_PROHIBITED_MEMORY_PHRASES = (
    "根据我的记忆",
    "根据记忆",
    "你上次说过",
    "数据库显示",
    "检索结果显示",
)

# 加载交互宪法模板（启动时一次性读取）
_SYSTEM_TEMPLATE: str = ""


def _load_system_template() -> str:
    """加载交互宪法模板。"""
    global _SYSTEM_TEMPLATE
    if not _SYSTEM_TEMPLATE:
        constitution_path = config.prompt_path("system_constitution.txt")
        if constitution_path.exists():
            _SYSTEM_TEMPLATE = constitution_path.read_text(encoding="utf-8")
        else:
            logger.warning("找不到 system_constitution.txt，使用内置最小模板")
            _SYSTEM_TEMPLATE = (
                "你是伴随用户共同成长的专属 AI 伙伴。\n"
                "- 严禁使用'根据我的记忆''数据库显示'等暴露记忆系统的表述。\n"
                "- 记忆应像潜意识一样自然影响你的回复。\n"
                "{procedural_memories}\n{semantic_memories}\n"
                "{episodic_memories}\n{emotional_memories}"
            )
    return _SYSTEM_TEMPLATE


def _build_system_prompt(context: ConstitutionalMemoryContext) -> str:
    """将记忆上下文填入交互宪法模板。"""
    template = _load_system_template()
    return template.format(
        procedural_memories=context.procedural_memories,
        semantic_memories=context.semantic_memories,
        episodic_memories=context.episodic_memories,
        emotional_memories=context.emotional_memories,
    )


def _warn_if_prohibited_memory_phrase(content: str) -> None:
    """记录暴露机械记忆系统的禁用表述。"""
    for phrase in _PROHIBITED_MEMORY_PHRASES:
        if phrase in content:
            logger.warning("回复包含禁用机械表述: %s", phrase)


async def generate_response(
    user_message: str,
    memory_context: ConstitutionalMemoryContext,
    conversation_history: list[dict[str, str]],
) -> str:
    """
    调用 LLM 生成带记忆上下文的回复。

    Args:
        user_message: 当前用户输入。
        memory_context: Constitutional AI 记忆上下文。
        conversation_history: 对话历史，每个元素含 {"role": ..., "content": ...}。

    Returns:
        str: 助手回复文本。

    Raises:
        httpx.HTTPError: API 调用失败时。
    """
    system_prompt = _build_system_prompt(memory_context)

    # 组装消息列表
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 只保留最近 N 条历史消息
    recent_history = conversation_history[-MAX_HISTORY_MESSAGES:]
    messages.extend(recent_history)

    # 追加当前用户消息
    messages.append({"role": "user", "content": user_message})

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                error_summary = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"Chat API 返回 {response.status_code}: {error_summary}",
                    request=response.request,
                    response=response,
                )

            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError(f"Chat API 响应中没有 choices: {data}")

            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                logger.warning("Chat API 返回了空 content")

            final_content = content.strip()
            _warn_if_prohibited_memory_phrase(final_content)
            return final_content

    except httpx.TimeoutException:
        raise httpx.TimeoutException(
            f"Chat API 请求超时（{REQUEST_TIMEOUT_SECONDS}s）"
        )
    except httpx.HTTPStatusError:
        raise
    except httpx.RequestError as e:
        raise httpx.RequestError(f"Chat API 请求失败: {e}") from e
