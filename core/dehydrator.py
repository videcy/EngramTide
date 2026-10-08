"""
Phase 4 — 会话脱水器（含话题分割）。

职责：
- 接收本轮会话历史。
- Phase 4：先调 LLM 做话题分割，再对每个话题片段独立压缩。
- 分割失败时整段回退到 Phase 3 行为（记忆零丢失）。
- 调用 LLM 进行脱水，输出 JSON 记忆数组。
- 解析 JSON，清洗字段。
- 为每条记忆计算 embedding。
- 构造 Memory 对象。
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

import config
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEHYDRATE_MAX_ITEMS,
    MAX_TOPIC_SEGMENTS,
    MIN_SEGMENT_MESSAGES,
    REQUEST_TIMEOUT_SECONDS,
    TOPIC_SPLIT_ENABLED,
)
from core.embedding import embed_text
from core.memory_store import Memory

logger = logging.getLogger(__name__)

# 允许的记忆类型
_ALLOWED_TYPES = {"semantic", "episodic", "emotional", "procedural"}



# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class SplitReport:
    """一次话题分割的摘要，供日志与测试断言。"""

    attempted: bool     # 是否尝试了分割（消息数达标且开关开启）
    segments: int       # 分割段数（回退时为 1）
    fell_back: bool     # 是否回退到整段脱水
    reason: str = ""    # 回退原因（LLM 失败 / JSON 非法 / 索引越界…）
    # 脱水失败段在输入消息里的下标区间 [start, end]（闭区间）。
    # 调用方据此只把成功段对应的原文标记为已处理。
    failed_ranges: list[tuple[int, int]] = field(default_factory=list)


# ── 工具函数 ──────────────────────────────────────────────


def _parse_bool(value: Any, default: bool = False) -> bool:
    """显式解析 LLM 返回的布尔字段。"""
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


def _load_prompt(filename: str) -> str:
    """加载 prompt 文件（PROMPTS_OVERRIDE_DIR 优先）。"""
    path = config.prompt_path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _format_conversation(messages: list[dict[str, str]]) -> str:
    """将对话历史格式化为可读文本。"""
    lines: list[str] = []
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _format_conversation_numbered(messages: list[dict[str, str]]) -> str:
    """将对话历史格式化为带编号的可读文本（供话题分割 prompt）。"""
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"  {i}: {role}: {msg['content']}")
    return "\n".join(lines)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """从 LLM 输出中提取第一个 JSON 数组。"""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
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
    """清洗单条记忆的字段。返回 None 表示该条应被丢弃。"""
    content = str(item.get("content", "")).strip()
    if not content:
        return None

    mem_type = str(item.get("type", "semantic")).strip().lower()
    if mem_type not in _ALLOWED_TYPES:
        mem_type = "semantic"

    try:
        valence = float(item.get("valence", 0.0))
    except (TypeError, ValueError):
        valence = 0.0
    valence = max(-1.0, min(1.0, valence))

    try:
        arousal = float(item.get("arousal", 0.0))
    except (TypeError, ValueError):
        arousal = 0.0
    arousal = max(0.0, min(1.0, arousal))

    unresolved = _parse_bool(item.get("unresolved", False))

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


# ── 话题分割 ──────────────────────────────────────────────


def _validate_segment_indices(
    segments: list[dict[str, int]],
    total_messages: int,
) -> str | None:
    """
    校验分割索引。返回 None 表示合法，否则返回错误原因字符串。

    规则：
    - 第一个 start == 0，最后一个 end == total_messages - 1
    - 相邻段 end + 1 == 下一段 start
    - 所有 start ≤ end
    - 无越界
    """
    if not segments:
        return "分割结果为空"

    for i, seg in enumerate(segments):
        start = seg.get("start", -1)
        end = seg.get("end", -1)
        if not isinstance(start, int) or not isinstance(end, int):
            return f"段 {i} 的 start/end 不是整数"
        if start < 0 or end >= total_messages:
            return f"段 {i} 索引越界: [{start},{end}]，消息总数 {total_messages}"
        if start > end:
            return f"段 {i} start > end: [{start},{end}]"

    if segments[0]["start"] != 0:
        return f"首段 start != 0: {segments[0]}"
    if segments[-1]["end"] != total_messages - 1:
        return f"末段 end != {total_messages - 1}: {segments[-1]}"

    for i in range(len(segments) - 1):
        if segments[i]["end"] + 1 != segments[i + 1]["start"]:
            return (
                f"段 {i} 与段 {i + 1} 有间隙或重叠: "
                f"[{segments[i]['start']},{segments[i]['end']}]"
                f" → [{segments[i+1]['start']},{segments[i+1]['end']}]"
            )

    return None


async def split_conversation(
    messages: list[dict[str, str]],
) -> tuple[list[list[dict[str, str]]], SplitReport]:
    """
    调 LLM 将对话按话题分割为消息子列表。

    规则：
    - TOPIC_SPLIT_ENABLED=False 或 len(messages) < MIN_SEGMENT_MESSAGES
      → 不调 LLM，整段返回。
    - LLM 输出经 _extract_json_array 解析；每段的 messages 索引必须
      合法（越界/重叠/遗漏 → 整体回退，宁可不分不可丢消息）。
    - 段数 > MAX_TOPIC_SEGMENTS → 超出部分并入最后一段 + warning。
    - 任何异常 → 整段回退 + warning。
    """
    if not TOPIC_SPLIT_ENABLED:
        return (
            [list(messages)],
            SplitReport(attempted=False, segments=1, fell_back=False),
        )

    if len(messages) < MIN_SEGMENT_MESSAGES:
        return (
            [list(messages)],
            SplitReport(attempted=False, segments=1, fell_back=False),
        )

    prompt_template = _load_prompt("topic_split.txt")
    if not prompt_template:
        logger.warning("topic_split.txt 未找到，跳过话题分割")
        return (
            [list(messages)],
            SplitReport(attempted=True, segments=1, fell_back=True, reason="prompt 缺失"),
        )

    conversation_text = _format_conversation_numbered(messages)
    user_prompt = (
        f"对话共有 {len(messages)} 条消息。\n\n"
        f"{conversation_text}\n\n"
        f"请输出 JSON 数组。"
    )

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": config.TOPIC_SPLIT_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                error_summary = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"话题分割 API 返回 {response.status_code}: {error_summary}",
                    request=response.request,
                    response=response,
                )
            data = response.json()
            raw_output = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("话题分割 LLM 调用失败: %s，整段回退", e)
        return (
            [list(messages)],
            SplitReport(attempted=True, segments=1, fell_back=True,
                        reason=f"LLM 调用失败: {e}"),
        )

    # 解析分割结果
    try:
        raw_segments = _extract_json_array(raw_output)
    except ValueError as e:
        logger.warning("话题分割 JSON 解析失败: %s，整段回退", e)
        return (
            [list(messages)],
            SplitReport(attempted=True, segments=1, fell_back=True,
                        reason=f"JSON 解析失败: {e}"),
        )

    # 校验索引
    err = _validate_segment_indices(raw_segments, len(messages))
    if err is not None:
        logger.warning("话题分割索引校验失败: %s，整段回退", err)
        return (
            [list(messages)],
            SplitReport(attempted=True, segments=1, fell_back=True,
                        reason=f"索引校验: {err}"),
        )

    # 按索引切分消息
    segments: list[list[dict[str, str]]] = []
    for seg in raw_segments:
        start = seg["start"]
        end = seg["end"]
        segments.append(messages[start:end + 1])

    # 段数超限：超出并入尾段
    if len(segments) > MAX_TOPIC_SEGMENTS:
        logger.warning(
            "话题分割段数 %d 超过 MAX_TOPIC_SEGMENTS=%d，合并尾部",
            len(segments),
            MAX_TOPIC_SEGMENTS,
        )
        merged = segments[MAX_TOPIC_SEGMENTS - 1:]
        flat: list[dict[str, str]] = []
        for s in merged:
            flat.extend(s)
        segments = segments[:MAX_TOPIC_SEGMENTS - 1] + [flat]

    return (segments, SplitReport(attempted=True, segments=len(segments), fell_back=False))


# ── 单段脱水（内部）───────────────────────────────────────


async def _dehydrate_segment(
    messages: list[dict[str, str]],
    source_conv_id: str,
) -> list[Memory]:
    """对单个话题片段执行脱水（与 Phase 3 dehydrate_conversation 的核心逻辑相同）。"""
    if not messages:
        return []

    prompt_template = _load_prompt("dehydrate.txt")
    if not prompt_template:
        prompt_template = (
            "从以下对话中提取有用信息，输出 JSON 数组。\n"
            "每条记忆: {{\"content\": \"...\", \"type\": \"semantic\", "
            "\"valence\": 0.0, \"arousal\": 0.0, \"unresolved\": false, \"tags\": []}}\n"
            "最多 {max_items} 条。"
        )

    conversation_text = _format_conversation(messages)
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
        "temperature": 0.3,
        "max_tokens": config.DEHYDRATE_MAX_TOKENS,
    }

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
        choice = data["choices"][0]
        raw_output = choice["message"]["content"] or ""

    if choice.get("finish_reason") == "length":
        logger.warning(
            "脱水输出被截断（finish_reason=length，DEHYDRATE_MAX_TOKENS=%d）；"
            "推理型模型的思考 token 也计入上限，可调大该值",
            config.DEHYDRATE_MAX_TOKENS,
        )

    raw_items = _extract_json_array(raw_output)
    if not raw_items:
        return []

    memories: list[Memory] = []
    for item in raw_items:
        cleaned = _clean_memory_item(item)
        if cleaned is None:
            continue
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

    return memories


# ── 公开入口 ──────────────────────────────────────────────


async def dehydrate_conversation(
    conversation: list[dict[str, str]],
    source_conv_id: str,
) -> tuple[list[Memory], SplitReport]:
    """
    将会话脱水为记忆列表（Phase 4：先话题分割再逐段脱水）。

    Args:
        conversation: 对话历史。
        source_conv_id: 本次会话 ID。

    Returns:
        (记忆对象列表, SplitReport)——SplitReport 供调用方打印分割信息（计划 §5.2）。
    """
    if not conversation:
        return ([], SplitReport(attempted=False, segments=0, fell_back=False))

    # Phase 4：话题分割
    segments, split_report = await split_conversation(conversation)

    if split_report.attempted and not split_report.fell_back:
        logger.info(
            "话题分割: 检测到 %d 个话题，分段压缩中...",
            split_report.segments,
        )
    elif split_report.fell_back:
        logger.info("话题分割回退: %s，整段压缩", split_report.reason)

    # 逐段脱水（逐段容错：单段失败跳过该段，不牵连其余段——记忆零丢失原则 §10）
    all_memories: list[Memory] = []
    failed_segments = 0
    last_error: Exception | None = None
    offset = 0  # 段是输入的连续切片（尾部合并后也是），累加长度即得下标
    for i, seg in enumerate(segments):
        seg_start = offset
        offset += len(seg)
        try:
            seg_memories = await _dehydrate_segment(seg, source_conv_id)
        except Exception as e:
            failed_segments += 1
            last_error = e
            split_report.failed_ranges.append((seg_start, offset - 1))
            logger.warning(
                "段 %d/%d 脱水失败: %s，跳过该段（其余段不受影响）",
                i + 1, len(segments), e,
            )
            continue
        all_memories.extend(seg_memories)

    # 全部段失败 → 维持 Phase 3 的失败语义，向上抛出由调用方统一报告
    if failed_segments == len(segments) and last_error is not None:
        raise last_error

    logger.info(
        "脱水完成：%d 段（失败 %d 段）→ %d 条有效记忆",
        split_report.segments,
        failed_segments,
        len(all_memories),
    )
    return (all_memories, split_report)


# ── 保留旧接口兼容（供测试直接调用）─────────────────────────

# Phase 4 不再保留独立的全量脱水入口——dehydrate_conversation 已内含分割步骤。
# 测试需要单段脱水时可调用 _dehydrate_segment。
